// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing

using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NNPP
{
    [System.Serializable]
    public class Conv2D : NNLayerBase
    {
        public int Filters;
        public int2 KernalSize;
        public int2 Stride;
        private ComputeBuffer outputbuffer;
        private ComputeBuffer weightbuffer;
        public Conv2D(KerasLayerConfigJson config) : base(config)
        {
            Filters = config.filters;
            KernalSize = new int2(config.kernel_size[0], config.kernel_size[1]);
            Stride = new int2(config.strides[0], config.strides[1]);
            KernelId = NNCompute.Instance.KernelConv2D(32);
        }

        public override void LoadWeight(KerasLayerWeightJson[] weightsKernel)
        {
            WeightShape = new int4(weightsKernel[0].shape[0],
                weightsKernel[0].shape[1],
                weightsKernel[0].shape[2],
                weightsKernel[0].shape[3]);
            int kernel_weight_length = WeightShape.x * WeightShape.y * WeightShape.z * WeightShape.w;
            int bias_weight_length = WeightShape.w;
            float[] Weights = new float[kernel_weight_length + bias_weight_length];
            for (int i = 0; i < WeightShape.x; i++)
            {
                for (int j = 0; j < WeightShape.y; j++)
                {
                    for (int k = 0; k < WeightShape.z; k++)
                    {
                        for (int w = 0; w < WeightShape.w; w++)
                        {
                            int arrayindex = i * WeightShape.y * WeightShape.z * WeightShape.w +
                                             j * WeightShape.z * WeightShape.w +
                                             k * WeightShape.w +
                                             w;
                            Weights[arrayindex] = weightsKernel[0].kernelweight[i, j, k, w];
                        }
                    }
                }
            }
            Array.Copy(weightsKernel[1].arrayweight, 0, Weights, kernel_weight_length, bias_weight_length);
            weightbuffer?.Release();
            weightbuffer = new ComputeBuffer(kernel_weight_length + bias_weight_length, sizeof(float));
            weightbuffer.SetData(Weights);
        }

        public override void Init(int4 inputShape)
        {
            InputShape = inputShape;
            OutputShape = inputShape;
            OutputShape.xy /= Stride;
            OutputShape.z = Filters;
            outputbuffer?.Release();
            outputbuffer = new ComputeBuffer(OutputShape.x * OutputShape.y * OutputShape.z, sizeof(float));
            int maxfilter = Mathf.Max(inputShape.z, Filters);
            KernelId = NNCompute.Instance.KernelConv2D(maxfilter);
            Output = outputbuffer;
        }

        public override void Release()
        {
            weightbuffer?.Release();
            outputbuffer?.Release();
        }

        public override void Run(object[] input, CommandBuffer cmd)
        {
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
            cmd.SetComputeBufferParam(NNCompute.Instance.Shader, KernelId, "Weights", weightbuffer);
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
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "WeightsShape", new int[4]
            {
                KernalSize.x,
                KernalSize.y,
                InputShape.z,
                Filters
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "WeightsShapeIdMultiplier", new int[4]
            {
                KernalSize.y * InputShape.z * Filters,
                InputShape.z * Filters,
                Filters,
                1
            });
            cmd.SetComputeIntParams(NNCompute.Instance.Shader, "Stride", new int[2]
            {
                Stride.x,
                Stride.y
            });
            //int group = Mathf.CeilToInt(OutputShape.z / 32.0f);

            cmd.DispatchCompute(NNCompute.Instance.Shader, KernelId, Mathf.CeilToInt(OutputShape.x / 4.0f), OutputShape.y, 1);
        }
    }
}