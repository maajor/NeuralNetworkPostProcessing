// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class OutputLayer : NNLayerBase
    {
        public RenderTexture outputTex;
        public OutputLayer(KerasLayerConfigJson config) : base(config)
        {
            KernelId = NNCompute.Instance.Kernel("OutputLayer");
        }

        public override void Run(object[] input)
        {
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer);
            NNCompute.Instance.Shader.SetTexture(KernelId, "OutputImage", outputTex);
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
            NNCompute.Instance.Shader.Dispatch(KernelId, OutputShape.x / 8, OutputShape.y / 8, 1);
        }

        public override void Init(Vector3Int inputShape)
        {
            base.Init(inputShape);
            if (outputTex != null)
                outputTex.Release();
            outputTex = new RenderTexture(OutputShape.y, OutputShape.x, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            outputTex.enableRandomWrite = true;
            outputTex.Create();
        }

        public override void Release()
        {
            if (outputTex != null)
                outputTex.Release();
        }
    }
}