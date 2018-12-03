using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Remoting.Messaging;
using Newtonsoft.Json;
using Unity.Mathematics;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using ComputeBuffer = UnityEngine.ComputeBuffer;

public class NeuralNetworkModel
{
    public List<NeuralNetworkLayer> Layers;
    private InputLayer Input;
    private OutputLayer Output;
    private CommandBuffer cb;

    public void Load(
        string architecturePath = "Model/model_architecture",
        string weightsPath = "Model/model_weight")
    {
        TextAsset architexturetext = Resources.Load<TextAsset>("Model/model_terrain");
        var modeljson = JsonConvert.DeserializeObject<KerasJson>(architexturetext.text);
        LoadModel(modeljson.model.config);
        LoadWeight(modeljson.weights);
    }

    private void LoadModel(KerasLayersJson layersJson)
    {
        Layers = new List<NeuralNetworkLayer>();
        Dictionary<string, List<string>> inputNodes = new Dictionary<string, List<string>>();
        foreach (var layer in layersJson.layers)
        {
            switch (layer.class_name)
            {
                case "InputLayer":
                    Input = new InputLayer(layer.config);
                    Layers.Add(Input);
                    break;
                case "Conv2D":
                    Layers.Add(new Conv2D(layer.config));
                    if (layer.config.activation == "relu")
                    {
                        Layers.Add(new ReLU(layer.config));
                    }
                    if (layer.config.activation == "tanh")
                    {
                        Layers.Add(new Tanh(layer.config));
                    }
                    break;
                case "LeakyReLU":
                    Layers.Add(new LeakyReLU(layer.config));
                    break;
                case "BatchNormalization":
                    Layers.Add(new BatchNormalization(layer.config));
                    break;
                case "UpSampling2D":
                    Layers.Add(new UpSampling2D(layer.config));
                    break;
                case "Concatenate":
                    var thislayer = new Concatenate(layer.config);
                    string alternativeLayerName = layer.inbound_nodes[0][1][0] as string;
                    thislayer.AlternativeInputId = Layers.FindIndex(ly => string.Compare(ly.Name, alternativeLayerName) == 0);
                    Layers.Add(thislayer);
                    break;
            }
        }
        Output = new OutputLayer(null);
    }

    private void LoadWeight(List<KerasLayerWeightJson> weights)
    {
        int weightcount = 0;
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is Conv2D)
            {
                Layers[i].LoadWeight(new KerasLayerWeightJson[2]
                {
                    weights[weightcount],
                    weights[weightcount + 1]
                });
                weightcount += 2;
            }
            if (Layers[i] is BatchNormalization)
            {
                Layers[i].LoadWeight(new KerasLayerWeightJson[4] {
                    weights[weightcount],
                    weights[weightcount + 1],
                    weights[weightcount + 2],
                    weights[weightcount + 3]
                });
                weightcount += 4;
            }
        }
    }

    public void Init(int height, int width)
    {
        Input.Init(new int4(height, width, 4, 0));
        //Layers[0].Init(new int4(height, width, 3, 0));
        for (int i = 1; i < Layers.Count; i++)
        {
            if (Layers[i] is Concatenate)
            {
                int4 input1 = Layers[i - 1].OutputShape;
                int4 input2 = Layers[(Layers[i] as Concatenate).AlternativeInputId].OutputShape;
                Layers[i].Init(new int4(input1.x, input1.y, input1.z + input2.z, 1));
            }
            else
            {
                Layers[i].Init(Layers[i - 1].OutputShape);
            }
        }
        Output.Init(Layers[Layers.Count - 1].OutputShape);
    }
    private int _height, _width;

    public void Setup(CommandBuffer cmd, RenderTargetIdentifier src, RenderTargetIdentifier dep, int height, int width)
    {
        if (_height != height || _width != width)
        {
            Init(height, width);
            _height = height;
            _width = width;
        }

        Input.src = src;
        Input.dep = dep;

        cb = cmd;
    }

    public RenderTexture Predict()
    {
        Input.Run(null, cb);
        for (int i = 1; i < Layers.Count; i++)
        {
            if (Layers[i] is Concatenate)
            {
                Layers[i].Run(new object[2]
                {
                    Layers[i - 1].Output,
                    Layers[(Layers[i] as Concatenate).AlternativeInputId].Output,
                }, cb);
            }
            else
            {
                Layers[i].Run(new object[1] {Layers[i - 1].Output}, cb);
            }
        }
        Output.Run(new object[1] { Layers[Layers.Count - 1].Output }, cb);
        return Output.outputTex;
    }

    public void Release()
    {
        foreach (var layer in Layers)
        {
            layer.Release();
        }
        Input.Release();
        Output.Release();
    }
}

[System.Serializable]
public class NeuralNetworkLayer
{
    public string Name;
    //public List<int> InputId;
    public int4 InputShape;
    public int4 OutputShape;
    public int4 WeightShape;
    public object Output;
    [SerializeField]
    protected int KernelId;
    public NeuralNetworkLayer(KerasLayerConfigJson config)
    {
        if (config != null)
            Name = config.name;
    }

    public virtual void LoadWeight(KerasLayerWeightJson[] weights)
    {

    }

    public virtual void Run(object[] input, CommandBuffer cmd)
    {
        Output = input[0];
    }

    public virtual void Init(int4 inputShape)
    {
        InputShape = inputShape;
        OutputShape = inputShape;
    }

    public virtual void Release()
    {
    }
}

public class InputLayer: NeuralNetworkLayer
{
    private ComputeBuffer outputbuffer;
    public RenderTargetIdentifier src;
    public RenderTargetIdentifier dep;
    public InputLayer(KerasLayerConfigJson config) : base(config)
    {
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("InputLayer");
    }

    public override void Run(object[] input, CommandBuffer cmd)
    {
        cmd.SetComputeTextureParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "InputImage", src);
        cmd.SetComputeTextureParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "InputImage1", dep);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShape", new int[3]
        {
            InputShape.x,
            InputShape.y,
            InputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShapeIdMultiplier", new int[3]
        {
            InputShape.y * InputShape.z,
            InputShape.z,
            1
        });
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, InputShape.x / 8, InputShape.y / 8, 1);
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
}
[System.Serializable]
public class Conv2D : NeuralNetworkLayer
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
        KernelId = NeuralNetworkComputeShader.Instance.KernelConv2D(32);
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
        KernelId = NeuralNetworkComputeShader.Instance.KernelConv2D(maxfilter);
        Output = outputbuffer;
    }

    public override void Release()
    {
        weightbuffer?.Release();
        outputbuffer?.Release();
    }

    public override void Run(object[] input, CommandBuffer cmd)
    {
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "Weights", weightbuffer);
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShape", new int[3]
        {
            InputShape.x,
            InputShape.y,
            InputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShapeIdMultiplier", new int[3]
        {
            InputShape.y * InputShape.z,
            InputShape.z,
            1
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "OutputShape", new int[3]
        {
            OutputShape.x,
            OutputShape.y,
            OutputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "OutputShapeIdMultiplier", new int[3]
        {
            OutputShape.y * OutputShape.z,
            OutputShape.z,
            1
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "WeightsShape", new int[4]
        {
            KernalSize.x,
            KernalSize.y,
            InputShape.z,
            Filters
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "WeightsShapeIdMultiplier", new int[4]
        {
            KernalSize.y * InputShape.z * Filters,
            InputShape.z * Filters,
            Filters,
            1
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "Stride", new int[2]
        {
            Stride.x,
            Stride.y
        });
        //int group = Mathf.CeilToInt(OutputShape.z / 32.0f);

        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x / 8, OutputShape.y , 1);
    }
}

public class ReLU : NeuralNetworkLayer
{
    protected ComputeBuffer outputbuffer;
    public ReLU(KerasLayerConfigJson config) : base(config)
    {
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("ReLU");
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
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x * OutputShape.y * OutputShape.z / 32, 1, 1);
    }
}

public class Tanh : NeuralNetworkLayer
{
    private ComputeBuffer outputbuffer;
    public Tanh(KerasLayerConfigJson config) : base(config)
    {
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("Tanh");
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
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x * OutputShape.y * OutputShape.z / 32, 1, 1);
    }
}

public class LeakyReLU : ReLU
{
    public float Alpha;
    public LeakyReLU(KerasLayerConfigJson config) : base(config)
    {
        Alpha = config.alpha;
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("LeakyReLU");
    }

    public override void Run(object[] input, CommandBuffer cmd)
    {
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.SetComputeFloatParam(NeuralNetworkComputeShader.Instance.Shader, "Alpha", Alpha);
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x * OutputShape.y * OutputShape.z / 32, 1, 1);
    }
}
[System.Serializable]
public class BatchNormalization : NeuralNetworkLayer
{
    private ComputeBuffer weightbuffer;
    private ComputeBuffer outputbuffer;
    public BatchNormalization(KerasLayerConfigJson config) : base(config)
    {
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("BatchNormalization");
    }

    public override void LoadWeight(KerasLayerWeightJson[] weightsKernel)
    {
        WeightShape.x = weightsKernel[0].shape[0];
        float[] Weights = new float[WeightShape.x * 4];
        for (int i = 0; i < WeightShape.x; i++)
        {
            Weights[i * 4]     = weightsKernel[0].arrayweight[i];
            Weights[i * 4 + 1] = weightsKernel[1].arrayweight[i];
            Weights[i * 4 + 2] = weightsKernel[2].arrayweight[i];
            Weights[i * 4 + 3] = weightsKernel[3].arrayweight[i];
        }
        weightbuffer?.Release();
        weightbuffer = new ComputeBuffer(WeightShape.x * 4, sizeof(float));
        weightbuffer.SetData(Weights);
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
        weightbuffer?.Release();
    }

    public override void Run(object[] input, CommandBuffer cmd)
    {
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "Weights", weightbuffer);
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShape", new int[3]
        {
            InputShape.x,
            InputShape.y,
            InputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "OutputShape", new int[3]
        {
            OutputShape.x,
            OutputShape.y,
            OutputShape.z
        });
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x * OutputShape.y / 32, OutputShape.z, 1);
    }
}

public class Concatenate : NeuralNetworkLayer
{
    private ComputeBuffer outputbuffer;
    public int AlternativeInputId;
    public Concatenate(KerasLayerConfigJson config) : base(config)
    {
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("Concatenate");
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
        var input0 = input[0] as ComputeBuffer;
        int inputfilters0 = input0.count / (OutputShape.x * OutputShape.y);
        int inputfilters1 = OutputShape.z - inputfilters0;
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input0);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput1", input[1] as ComputeBuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShape", new int[3]
        {
            InputShape.x,
            InputShape.y,
            inputfilters0
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShapeIdMultiplier", new int[3]
        {
            InputShape.y * inputfilters0,
            inputfilters0,
            1
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShapeIdMultiplier1", new int[3]
        {
            InputShape.y * inputfilters1,
            inputfilters1,
            1
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "OutputShape", new int[3]
        {
            OutputShape.x,
            OutputShape.y,
            OutputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "OutputShapeIdMultiplier", new int[3]
        {
            OutputShape.y * OutputShape.z,
            OutputShape.z,
            1
        });
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x / 8, OutputShape.y / 8, OutputShape.z);
        //Output = input0;
    }
}

public class UpSampling2D : NeuralNetworkLayer
{
    public int2 Size;
    private ComputeBuffer outputbuffer;
    public UpSampling2D(KerasLayerConfigJson config) : base(config)
    {
        Size = new int2(config.size[0], config.size[1]);
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("UpSampling2D");
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
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerOutput", outputbuffer);
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShape", new int[3]
        {
            InputShape.x,
            InputShape.y,
            InputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShapeIdMultiplier", new int[3]
        {
            InputShape.y * InputShape.z,
            InputShape.z,
            1
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "OutputShape", new int[3]
        {
            OutputShape.x,
            OutputShape.y,
            OutputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "OutputShapeIdMultiplier", new int[3]
        {
            OutputShape.y * OutputShape.z,
            OutputShape.z,
            1
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "Size", new int[2]
        {
            Size.x,
            Size.y
        });
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x / 8, OutputShape.y / 8, OutputShape.z);
    }
}

public class OutputLayer : NeuralNetworkLayer
{
    public RenderTexture outputTex;
    public OutputLayer(KerasLayerConfigJson config) : base(config)
    {
        KernelId = NeuralNetworkComputeShader.Instance.Kernel("OutputLayer");
    }

    public override void Run(object[] input, CommandBuffer cmd)
    {
        cmd.SetComputeBufferParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "LayerInput0", input[0] as ComputeBuffer);
        cmd.SetComputeTextureParam(NeuralNetworkComputeShader.Instance.Shader, KernelId, "OutputImage", outputTex);
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShape", new int[3]
        {
            InputShape.x,
            InputShape.y,
            InputShape.z
        });
        cmd.SetComputeIntParams(NeuralNetworkComputeShader.Instance.Shader, "InputShapeIdMultiplier", new int[3]
        {
            InputShape.y * InputShape.z,
            InputShape.z,
            1
        });
        cmd.DispatchCompute(NeuralNetworkComputeShader.Instance.Shader, KernelId, OutputShape.x / 8, OutputShape.y / 8, 1);
    }

    public override void Init(int4 inputShape)
    {
        base.Init(inputShape);
        outputTex?.Release();
        outputTex = new RenderTexture(OutputShape.y, OutputShape.x, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        outputTex.enableRandomWrite = true;
        outputTex.Create();
    }

    public override void Release()
    {
        outputTex?.Release();
    }
}

public class NeuralNetworkComputeShader
{
    private static NeuralNetworkComputeShader _instance;

    public static NeuralNetworkComputeShader Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = new NeuralNetworkComputeShader();
                _instance.Init();
            }

            return _instance;
        }

    }

    private int[] Conv2DKernelLayers = new int[6] {8,11,16,19,32,64};
    private int[] Conv2DKernels = new int[6];

    public ComputeShader Shader;
    private string shaderpath = "NeuralNetworkLayer";
    private int LeakyReluKernel, BatchNormalizationKernel, InputLayerKernel, 
        ConcatenateKernel, OutputLayerKernel, UpSampling2DKernel, ReluKernel, TanhKernel;

    private void Init()
    {
        Conv2DKernels = new int[6];
        Shader = Resources.Load<ComputeShader>(shaderpath);
        for (int i = 0; i < Conv2DKernelLayers.Length; i++)
        {
            Conv2DKernels[i] = Shader.FindKernel(string.Format("Conv2D_{0}", Conv2DKernelLayers[i]));
        }
        LeakyReluKernel = Shader.FindKernel("LeakyReLU");
        BatchNormalizationKernel = Shader.FindKernel("BatchNormalization");
        InputLayerKernel = Shader.FindKernel("InputLayer");
        OutputLayerKernel = Shader.FindKernel("OutputLayer");
        UpSampling2DKernel = Shader.FindKernel("UpSampling2D");
        ConcatenateKernel = Shader.FindKernel("Concatenate");
        ReluKernel = Shader.FindKernel("ReLU");
        TanhKernel = Shader.FindKernel("Tanh");
    }

    public int KernelConv2D(int channel)
    {
        for (int i = 0; i < Conv2DKernelLayers.Length; i++)
        {
            if (channel <= Conv2DKernelLayers[i])
            {
                return Conv2DKernels[i];
            }
        }
        return -1;
    }

    public int Kernel(string name)
    {
        switch (name)
        {
            case ("LeakyReLU"):
                return LeakyReluKernel;
            case ("BatchNormalization"):
                return BatchNormalizationKernel;
            case ("InputLayer"):
                return InputLayerKernel;
            case ("OutputLayer"):
                return OutputLayerKernel;
            case ("UpSampling2D"):
                return UpSampling2DKernel;
            case ("ReLU"):
                return ReluKernel;
            case ("Tanh"):
                return TanhKernel;
            case ("Concatenate"):
                return ConcatenateKernel;
            default:
                return -1;
        }
    }
}