// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class UpSampling2D : NNLayerBase
    {
        public int2 Size;
        private ComputeBuffer outputbuffer;
        public UpSampling2D(KerasLayerConfigJson config) : base(config)
        {
            Size = new int2(config.size[0], config.size[1]);
            KernelId = NNCompute.Instance.Kernel("UpSampling2D");
        }
        public override void Init(int4 inputShape)
        {
            InputShape = inputShape;
            OutputShape = inputShape;
            OutputShape.xy *= Size;
            outputbuffer?.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            Output = outputbuffer;
        }

        public override void Release()
        {
            outputbuffer?.Release();
        }

        public override void Run(object[] input, CommandBuffer cmd)
        {
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
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
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "OutputShape", new int[3]
            {
                OutputShape.x,
                OutputShape.y,
                OutputShape.z
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "OutputShapeIdMultiplier", new int[3]
            {
                OutputShape.y * OutputShape.z,
                OutputShape.z,
                1
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "Size", new int[2]
            {
                Size.x,
                Size.y
            });
            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, Mathf.CeilToInt(OutputShape.x / 8.0f), Mathf.CeilToInt(OutputShape.y / 8.0f), OutputShape.z);
        }
    }
}