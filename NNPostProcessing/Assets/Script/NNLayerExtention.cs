using System;
using System.Collections;
using System.Collections.Generic;
using NNPP;
using UnityEngine;

namespace NNPPUtil
{
    public static class NNLayerExtention  {

        public static void LoadWeight(this BatchNormalization layer, KerasLayerWeightJson[] weightsKernel)
        {
            layer.WeightShape.x = weightsKernel[0].shape[0];
            layer.weightcache = new float[(int)layer.WeightShape.x * 4];
            for (int i = 0; i < layer.WeightShape.x; i++)
            {
                layer.weightcache[i * 4] = weightsKernel[0].arrayweight[i];
                layer.weightcache[i * 4 + 1] = weightsKernel[1].arrayweight[i];
                layer.weightcache[i * 4 + 2] = weightsKernel[2].arrayweight[i];
                layer.weightcache[i * 4 + 3] = weightsKernel[3].arrayweight[i];
            }
        }

        public static void LoadWeight(this Conv2D layer, KerasLayerWeightJson[] weightsKernel)
        {
            Vector4 WeightShape = new Vector4(weightsKernel[0].shape[0],
                weightsKernel[0].shape[1],
                weightsKernel[0].shape[2],
                weightsKernel[0].shape[3]);
            layer.WeightShape = WeightShape;
            int kernel_weight_length = (int)(WeightShape.x * WeightShape.y * WeightShape.z * WeightShape.w);
            int bias_weight_length = (int)WeightShape.w;
            layer.weightcache = new float[kernel_weight_length + bias_weight_length];
            for (int i = 0; i < WeightShape.x; i++)
            {
                for (int j = 0; j < WeightShape.y; j++)
                {
                    for (int k = 0; k < WeightShape.z; k++)
                    {
                        for (int w = 0; w < WeightShape.w; w++)
                        {
                            float arrayindex = i * WeightShape.y * WeightShape.z * WeightShape.w +
                                               j * WeightShape.z * WeightShape.w +
                                               k * WeightShape.w +
                                               w;
                            layer.weightcache[(int)arrayindex] = weightsKernel[0].kernelweight[i, j, k, w];
                        }
                    }
                }
            }
            Array.Copy(weightsKernel[1].arrayweight, 0, layer.weightcache, kernel_weight_length, bias_weight_length);
        }

        public static void LoadConfig(this Conv2D layer, KerasLayerConfigJson config)
        {
            layer.Filters = config.filters;
            layer.KernalSize = new Vector2Int(config.kernel_size[0], config.kernel_size[1]);
            layer.Stride = new Vector2Int(config.strides[0], config.strides[1]);
        }

        public static void LoadConfig(this InputLayer layer, KerasLayerConfigJson config)
        {
            layer.InputChannels = int.Parse(config.batch_input_shape[3]);
        }

        public static void LoadConfig(this LeakyReLU layer, KerasLayerConfigJson config)
        {
            layer.Alpha = config.alpha;
        }

        public static void LoadConfig(this UpSampling2D layer, KerasLayerConfigJson config)
        {
            layer.Size = new Vector2Int(config.size[0], config.size[1]);
        }
    }
}
