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
        public RenderTargetIdentifier src;
        public RenderTargetIdentifier dep;
        public InputLayer(KerasLayerConfigJson config) : base(config)
        {
            KernelId = NNCompute.Instance.Kernel("InputLayer");
            InputChannels = int.Parse(config.batch_input_shape[3]);
        }

        public override void Run(object[] input, CommandBuffer cmd)
        {
            cmd.SetComputeTextureParam(NNCompute.Instance.Shader, KernelId, "InputImage", src);
            cmd.SetComputeTextureParam(NNCompute.Instance.Shader, KernelId, "InputImage1", dep);
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "InputShape", new int[3]
            {
                InputShape.x,
                InputShape.y,
                InputShape.z
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "InputShapeIdMultiplier", new int[3]
            {
                InputShape.y * InputShape.z,
                InputShape.z,
                1
            });
            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, Mathf.CeilToInt(InputShape.x / 8.0f), Mathf.CeilToInt(InputShape.y / 8.0f), 1);
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