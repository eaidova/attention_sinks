
from argparse import ArgumentParser
from pathlib import Path
from optimum.exporters.openvino import export_models
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.exporters.onnx.__main__ import _get_submodels_and_onnx_configs as _get_submodels_and_export_configs
from attention_sinks import AutoModelForCausalLM as ASAutoModelForCausalLM
from transformers import AutoTokenizer


def convert_optimum(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    out_dir = Path(model_id.split("/")[-1])
    model = ASAutoModelForCausalLM.from_pretrained(model_id, attention_sink_window_size=100)
    dummy_shapes = DEFAULT_DUMMY_SHAPES
    _, models_and_onnx_configs = _get_submodels_and_export_configs(
        model=model,
        task="text-generation-with-past",
        custom_onnx_configs={},
        custom_architecture=None,
        fn_get_submodels=None,
        preprocessors=None,
        _variant="default",
        monolith=False
    )
    if "decoder_with_past_model" in models_and_onnx_configs:
        models_and_onnx_configs = {"model": models_and_onnx_configs["decoder_with_past_model"]}
    ov_out_dir = Path(out_dir)
    model.config.save_pretrained(ov_out_dir)
    files_subpaths = ["openvino_" + model_name + ".xml" for model_name in models_and_onnx_configs.keys()]
    export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        output_dir=ov_out_dir,
        output_names=files_subpaths,
        input_shapes=dummy_shapes,
        device="cpu",
        fp16=True,
        int8=False,
        model_kwargs={},
    )
    tok.save_pretrained(ov_out_dir)

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="model id or path", default="databricks/dolly-v2-3b")

    args = parser.parse_args()
    convert_optimum(args.model)

main()