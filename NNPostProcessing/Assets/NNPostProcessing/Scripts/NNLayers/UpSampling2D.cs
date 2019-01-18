// neural network post-processing

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    public class UpSampling2D : NNLayerBase
    {
        public Vector2Int Size;
        private ComputeBuffer outputbuffer;
        public UpSampling2D() : base()
        {
            KernelId = NNCompute.Instance.Kernel("UpSampling2D");
        }
        public override void Init(Vector3Int inputShape)
        {
            InputShape = inputShape;
            OutputShape = new Vector3Int(inputShape.x * Size.x, inputShape.y * Size.y, inputShape.z);
            if (outputbuffer != null)
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
            NNCompute.Instance.Shader.SetBuffer(KernelId, "LayerInput0", input[0] as ComputeBuffer);
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
            NNCompute.Instance.Shader.SetInts("Size", new int[2]
            {
                Size.x,
                Size.y
            });
            NNCompute.Instance.Shader.Dispatch(KernelId, Mathf.CeilToInt(OutputShape.x / 8.0f), Mathf.CeilToInt(OutputShape.y / 8.0f), OutputShape.z);
        }
    }
}