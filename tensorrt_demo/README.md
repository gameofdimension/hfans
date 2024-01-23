# the tensorrt pipeline demo. https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#ex-deploy-onnx

0. `pip install pycuda tensorrt -i https://pypi.tuna.tsinghua.edu.cn/simple`
1. export model to onnx format. `python export.py`
2. build tensorrt engine. `sh build_engine.sh`
3. run pipeline. `python pipeline.py`