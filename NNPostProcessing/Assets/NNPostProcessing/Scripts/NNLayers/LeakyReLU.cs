// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class LeakyReLU : ReLU
    {
        public float Alpha;
        public LeakyReLU(KerasLayerConfigJson config) : base(config)
        {
            Alpha = config.alpha;
            KernelId = NNCompute.Instance.Kernel("LeakyReLU");
        }

        public override void Run(object[] input, CommandBuffer cmd)
        {
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
            cmd.SetComputeFloatParam(NNCompute.Instance.Shader, "Alpha", Alpha);
            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, Mathf.CeilToInt(OutputShape.x * OutputShape.y * OutputShape.z / 32.0f), 1, 1);
        }
    }
}