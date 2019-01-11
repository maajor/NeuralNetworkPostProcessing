// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class Add : NNLayerBase
    {
        private ComputeBuffer outputbuffer;
        public int AlternativeInputId;
        public Add(KerasLayerConfigJson config) : base(config)
        {
            KernelId = NNCompute.Instance.Kernel("Add");
        }

        public override void Init(Vector3Int inputShape)
        {
            base.Init(inputShape);
            if(outputbuffer != null)
                outputbuffer.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            Output = outputbuffer;
        }

        public override void Release()
        {
            if (outputbuffer != null)
                outputbuffer.Release();
        }

        public override void Run(object[] input)
        {
            var input0 = input[0] as ComputeBuffer;
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input0);
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput1", input[1] as ComputeBuffer);
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerOutput", outputbuffer);
            NNCompute.Instance.Shader.SetInts("InputShape", new int[3]
            {
                (int)InputShape.x,
                InputShape.y,
                InputShape.z
            });
            NNCompute.Instance.Shader.SetInts("InputShapeIdMultiplier", new int[3]
            {
                InputShape.y * InputShape.z,
                InputShape.z,
                1
            });
            NNCompute.Instance.Shader.SetInts("OutputShape", new int[3]
            {
                OutputShape.x,
                OutputShape.y,
                OutputShape.z
            });
            NNCompute.Instance.Shader.SetInts("OutputShapeIdMultiplier", new int[3]
            {
                OutputShape.y * OutputShape.z,
                OutputShape.z,
                1
            });
            NNCompute.Instance.Shader.Dispatch(KernelId, Mathf.CeilToInt(OutputShape.x / 8.0f), Mathf.CeilToInt(OutputShape.y / 8.0f), OutputShape.z);
        }
    }
}
