// neural network post-processing
// https://github.com/maajor/NeuralNetworkPostProcessing
//#define DEBUG_LAYER

using System;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.Rendering;


namespace NNPP
{
    public class NNModel
    {
        public List<NNLayerBase> Layers;
        private InputLayer Input;
        private OutputLayer Output;
        private CommandBuffer cb;
        public int debug_layer = 0;

        public void Load(string modelname)
        {
            TextAsset architexturetext = Resources.Load<TextAsset>("Model/" + modelname);
            var modeljson = JsonConvert.DeserializeObject<KerasJson>(architexturetext.text);
            LoadModel(modeljson.model.config);
            LoadWeight(modeljson.weights);
        }

        private void LoadModel(KerasLayersJson layersJson)
        {
            Layers = new List<NNLayerBase>();
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
            Input.Init(new Vector3Int(height, width, Input.InputChannels));
#if !DEBUG_LAYER
            for (int i = 1; i < Layers.Count; i++)
#else
            for (int i = 1; i < debug_layer + 1; i++)
#endif
            {
                if (Layers[i] is Concatenate)
                {
                    Vector3Int input1 = Layers[i - 1].OutputShape;
                    Vector3Int input2 = Layers[(Layers[i] as Concatenate).AlternativeInputId].OutputShape;
                    Layers[i].Init(new Vector3Int(input1.x, input1.y, input1.z + input2.z));
                }
                else
                {
                    Layers[i].Init(Layers[i - 1].OutputShape);
                }
            }
#if !DEBUG_LAYER
            Output.Init(Layers[Layers.Count - 1].OutputShape);
#else
            Output.Init(Layers[debug_layer].OutputShape);
#endif
        }
        private int _height, _width;

        private void Setup(RenderTexture src)
        {
            if (_height != src.height || _width != src.width)
            {
                Init(src.height, src.width);
                _height = src.height;
                _width = src.width;
            }
            Input.src = src;
        }

        public RenderTexture Predict(RenderTexture src)
        {
            Setup(src);
            Input.Run(null);
#if !DEBUG_LAYER
            for (int i = 1; i < Layers.Count; i++)
#else
            for (int i = 1; i < debug_layer + 1; i++)
#endif
            {
                if (Layers[i] is Concatenate)
                {
                    Layers[i].Run(new object[2]
                    {
                        Layers[i - 1].Output,
                        Layers[(Layers[i] as Concatenate).AlternativeInputId].Output,
                    });
                }
                if (Layers[i] is Add)
                {
                    Layers[i].Run(new object[2]
                    {
                        Layers[i - 1].Output,
                        Layers[(Layers[i] as Add).AlternativeInputId].Output,
                    });
                }
                else
                {
                    Layers[i].Run(new object[1] {Layers[i - 1].Output});
                }
            }
#if !DEBUG_LAYER
            Output.Run(new object[1] { Layers[Layers.Count - 1].Output });
#else
            Output.Run(new object[1] { Layers[debug_layer].Output });
#endif
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