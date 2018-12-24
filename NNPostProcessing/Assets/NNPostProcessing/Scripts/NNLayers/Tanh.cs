// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class Tanh : NNLayerBase
    {
        private ComputeBuffer outputbuffer;
        public Tanh(KerasLayerConfigJson config) : base(config)
        {
            KernelId = NNCompute.Instance.Kernel("Tanh");
        }

        public override void Init(int4 inputShape)
        {
            base.Init(inputShape);
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
            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, Mathf.CeilToInt(OutputShape.x * OutputShape.y * OutputShape.z / 32.0f), 1, 1);
        }
    }
}