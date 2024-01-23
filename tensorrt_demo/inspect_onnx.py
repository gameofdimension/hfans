import onnx


def main():
    model = onnx.load('build/resnet50_onnx_model.onnx')
    output = [node.name for node in model.graph.output]

    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))

    print('Inputs: ', net_feed_input)
    print('Outputs: ', output)


if __name__ == '__main__':
    main()
