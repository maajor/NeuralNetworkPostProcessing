// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class InputLayer : NNLayerBase
    {
        private ComputeBuffer outputbuffer;
        public int InputChannels;
        public RenderTexture src;
        public InputLayer(KerasLayerConfigJson config) : base(config)
        {
            KernelId = NNCompute.Instance.Kernel("InputLayer");
            InputChannels = int.Parse(config.batch_input_shape[3]);
        }

        public override void Run(object[] input)
        {

            NNCompute.Instance.Shader.SetTexture(KernelId, "InputImage", src);
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);
            NNCompute.Instance.Shader.SetInts("InputShape", new int[3]
            {
                InputShape.x,
                InputShape.y,
                InputShape.z
            });
            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier", new int[3]
            {
                InputShape.y * InputShape.z,
                InputShape.z,
                1
            });
            NNCompute.Instance.Shader.Dispatch(KernelId, Mathf.CeilToInt(InputShape.x / 8.0f), Mathf.CeilToInt(InputShape.y / 8.0f), 1);
        }

        public override void Init(Vector3Int inputShape)
        {
            base.Init(inputShape);
            if(outputbuffer !=null)
                outputbuffer.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            Output = outputbuffer;
        }

        public override void Release()
        {
            if (outputbuffer != null)
                outputbuffer.Release();
        }
    }
}