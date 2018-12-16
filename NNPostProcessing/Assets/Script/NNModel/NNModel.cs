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

namespace NNPP
{
    public class NNModel
    {
        public List<NNLayerBase> Layers;
        private InputLayer Input;
        private OutputLayer Output;
        private CommandBuffer cb;

        public void Load()
        {
            TextAsset architexturetext = Resources.Load<TextAsset>("Model/NNmodel");
            var modeljson = JsonConvert.DeserializeObject<KerasJson>(architexturetext.text);
            LoadModel(modeljson.model.config);
            LoadWeight(modeljson.weights);
        }

        private void LoadModel(KerasLayersJson layersJson)
        {
            Layers = new List<NNLayerBase>();
            Dictionary<string, List<string>> inputNodes = new Dictionary<string, List<string>>();
            foreach (var layer in layersJson.layers)
            {
                switch (layer.class_name)
                {
                    case "InputLayer":
                        Input = new InputLayer(layer.config);
                        Layers.Add(Input);
                        break;
                    case "Activation":
                        if (layer.config.activation == "relu")
                        {
                            Layers.Add(new ReLU(layer.config));
                        }
                        if (layer.config.activation == "tanh")
                        {
                            Layers.Add(new Tanh(layer.config));
                        }
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
                    case "Add":
                        var addlayer = new Add(layer.config);
                        int alterinput = layer.inbound_nodes[0].FindIndex(node =>
                            string.Compare(node[0] as string, Layers[Layers.Count - 1].Name) != 0);
                        string addalternativeLayerName = layer.inbound_nodes[0][alterinput][0] as string;
                        addlayer.AlternativeInputId = Layers.FindIndex(ly => string.Compare(ly.Name, addalternativeLayerName) == 0);
                        Layers.Add(addlayer);
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
            //Output.Init(Layers[31].OutputShape);
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
                if (Layers[i] is Add)
                {
                    Layers[i].Run(new object[2]
                    {
                        Layers[i - 1].Output,
                        Layers[(Layers[i] as Add).AlternativeInputId].Output,
                    }, cb);
                }
                else
                {
                    Layers[i].Run(new object[1] {Layers[i - 1].Output}, cb);
                }
            }
            Output.Run(new object[1] { Layers[Layers.Count - 1].Output }, cb);
            //Output.Run(new object[1] { Layers[31].Output }, cb);
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
}