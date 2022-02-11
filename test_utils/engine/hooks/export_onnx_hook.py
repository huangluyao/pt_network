import torch
import os
from .hook import Hook, HOOKS
from test_utils.utils.checkpoint import load_checkpoint


@HOOKS.registry()
class ExportOnnxHook(Hook):

    def __init__(self, input_name, output_name, dynamic_axes=None, opset_version=12, output_file=None):
        super(ExportOnnxHook, self).__init__()
        self.input_name = input_name
        self.output_name = output_name
        self.opset_version = opset_version
        self.output_file = output_file
        if dynamic_axes is not None:
            self.dynamic_axes = dynamic_axes
        else:
            self.dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"}
        }
    def after_train(self, runner):
        runner.logger.info("export onnx ...")
        model_path = os.path.join(runner.cfg.output_dir, "checkpoints", "val_best.pth")
        model = runner.model.to("cpu")
        if os.path.exists(model_path):
            load_checkpoint(model, model_path)

        if self.output_file is None:
            self.output_file = model_path.replace("pth", "onnx")

        input_size = runner.cfg.input_size
        fake_data = torch.randn([1, 3]+input_size, device="cpu")
        torch.onnx.export(model, (fake_data), self.output_file,
                          input_names=[self.input_name],
                          output_names=[self.output_name],
                          dynamic_axes=self.dynamic_axes,
                          opset_version=self.opset_version,
                          )
        runner.logger.info("export onnx success, onnx file at {}".format(self.output_file))


