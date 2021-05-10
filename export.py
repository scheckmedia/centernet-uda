import argparse
from pathlib import Path
import torch
import torch.nn as nn
from backends.decode import decode_detection
import yaml
from importlib import import_module

try:
    import onnx
    import onnx.utils
    from onnxsim import simplify
    has_simplify = True
except BaseException:
    has_simplify = False
    pass


class CenterNet(nn.Module):
    def __init__(self, backend, max_detections, is_rotated=False, nms=3):
        super().__init__()
        self.backend = backend
        self.max_detections = max_detections
        self.is_rotated = is_rotated
        self.nms = nms

    def forward(self, x):
        out = self.backend(x)
        has_kps = "kps" in out

        dets = decode_detection(
            torch.clamp(out["hm"].sigmoid_(), min=1e-4, max=1 - 1e-4),
            out["wh"],
            out["reg"],
            kps=out["kps"] if has_kps else None,
            K=self.max_detections,
            rotated=self.is_rotated,
            nms_size=self.nms)

        if has_kps:
            dets, kps = dets
            kps[..., 0:2] *= self.backend.down_ratio

        dets[:, :, :4] *= self.backend.down_ratio

        # boxes, scores, classes
        if self.is_rotated:
            if has_kps:
                return dets[:, :, :5], dets[:, :, 5], dets[:, :, 6], kps

            return dets[:, :, :5], dets[:, :, 5], dets[:, :, 6]

        if has_kps:
            return dets[:, :, :4], dets[:, :, 4], dets[:, :, 5], kps

        return dets[:, :, :4], dets[:, :, 4], dets[:, :, 5]


def build_model(experiment, model_spec, without_decode_detections,
                max_detections, nms=3, use_last=True):

    module = import_module(f"backends.{model_spec['name']}")
    backend = getattr(module, 'build')(**model_spec["params"])

    ckpt = experiment / ('model_last.pth' if use_last else 'model_best.pth')
    if ckpt.exists():
        checkpoint = torch.load(ckpt)
        backend.load_state_dict(checkpoint['state_dict'])
        print(f"Restore weights {ckpt} successful!")
    else:
        print(f"No weights were found in folder {experiment}")

    if model_spec['name'] == 'efficientnet':
        backend.base.set_swish(memory_efficient=False)

    if not without_decode_detections:
        model = CenterNet(backend, max_detections,
                          model_spec["params"]["rotated_boxes"], nms)
    else:
        model = backend

    model.eval()
    return model


def export_model(experiment, model, model_name,
                 input_shape, without_decode_detections, simplify_model):
    shape = [1, ] + input_shape
    x = torch.randn(*shape, requires_grad=True)
    torch_out = model(x)
    possible_outputs = ['boxes', 'scores', 'classes', 'kps']

    if without_decode_detections:
        outputs = list(torch_out.keys())
    else:
        outputs = [possible_outputs[i] for i in range(len(torch_out))]

    suffix = '_wd' if without_decode_detections else ''
    output_path = experiment / \
        f"centernet_{model_name}_{shape[2]}x{shape[3]}{suffix}.onnx"
    torch.onnx.export(model,               # model being run
                      x,
                      output_path,
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=outputs)
    # dynamic_axes={'input': {0: 'batch_size'}})

    print(f"Export model to {output_path} successful!")

    if simplify_model:
        print("Simplify model")
        if not has_simplify:
            print("Can't simplify model because ONNX Simplifier is not install")
            print("Please install it: pip install onnx-simplifier")
            return

        onnx_model = onnx.load(output_path)
        simplyfied_model, check = simplify(
            onnx_model, input_shapes={"input": shape})
        simplyfied_model = onnx.shape_inference.infer_shapes(simplyfied_model)
        onnx_model = onnx.utils.polish_model(simplyfied_model)
        onnx.checker.check_model(simplyfied_model)
        onnx.checker.check_model(onnx_model)

        output_path = experiment / \
            f"centernet_{model_name}_{shape[2]}x{shape[3]}{suffix}_smpl.onnx"
        onnx.save(simplyfied_model, output_path)

        print(f"Export simplified model to {output_path} successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export an experiment to onnx model')

    parser.add_argument(
        "-e", "--experiment", required=True,
        help="Path to the experiment that should be exported.")
    parser.add_argument(
        "-i",
        "--input_shape",
        type=int,
        nargs='+',
        default=[800, 800],
        help="Input size for onnx model")
    parser.add_argument(
        "-l", "--use_last", action="store_true",
        help="If given, not the best but the last checkpoint will be restored.")
    parser.add_argument(
        "-wd", "--without_decode_detections", action="store_true",
        help="If given, CenterNet output will be the heads instead of boxes. \
            The reason for this is that boxes leads to problems with TensorRT.")
    parser.add_argument(
        "-s", "--simplify", action="store_true",
        help="If given it will be tried to simplify the model using onnx-simplifier."
    )
    parser.add_argument(
        "--nms", type=int, default=3,
        help="Kernel size for max pooling or CenterNet NMS."
    )

    args = parser.parse_args()
    experiment = Path(args.experiment)

    if not experiment.exists():
        print("Experiment does not exists!")
        exit(-1)

    shape_len = len(args.input_shape)
    if shape_len == 1:
        input_shape = [3, args.input_shape[0], args.input_shape[1]]
    elif shape_len == 2:
        input_shape = [3, ] + args.input_shape
    else:
        print("Invlaid input shape ")
        exit(-2)

    g = list(experiment.glob('*/config.yaml'))
    if len(g) == 0:
        print("No config.yaml file where found in experiment folder!")
        exit(-1)

    with open(g[0]) as f:
        cfg = yaml.load(f)
        model_specs = cfg["model"]["backend"]

    model = build_model(
        experiment,
        model_specs,
        args.without_decode_detections,
        cfg["max_detections"],
        args.nms,
        args.use_last)

    export_model(
        experiment,
        model,
        model_specs["name"],
        input_shape,
        args.without_decode_detections,
        args.simplify)
