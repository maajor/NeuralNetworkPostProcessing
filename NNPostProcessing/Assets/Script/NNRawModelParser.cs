using System;
using System.Collections;
using System.Collections.Generic;
using NNPP;
using UnityEditor;
using UnityEngine;
using Newtonsoft.Json;

namespace NNPPUtil
{
    public class NNRawModelParser : MonoBehaviour
    {
        [MenuItem("Assets/ParseFromRawModel")]
        public static void Parse()
        {
            TextAsset tex = Selection.activeObject as TextAsset;
            
            var allLayers = ParseAllLayers(tex.text);
            NNModelSerialize modeljson = new NNModelSerialize();
            foreach (var layer in allLayers)
            {
                object derivetype = Convert.ChangeType(layer, layer.GetType());
                string value = JsonUtility.ToJson(derivetype, true);
                modeljson.LayerTypes.Add(layer.GetType().FullName);
                modeljson.LayerJson.Add(value);
            }

            string serialized = JsonUtility.ToJson(modeljson);
            string outpath = Application.dataPath + "/NNPostProcessing/Resources/Model/" + tex.name + ".json";
            Debug.Log(outpath);
            System.IO.File.WriteAllText(outpath, serialized);

            foreach (var layer in allLayers)
            {
                layer.Release();
            }
        }

        public static List<NNLayerBase> ParseAllLayers(string text)
        {
            List<NNLayerBase> Layers = new List<NNLayerBase>();
            var modeljson = JsonConvert.DeserializeObject<KerasJson>(text);
            LoadModel(modeljson.model.config, Layers);
            LoadWeight(modeljson.weights, Layers);
            return Layers;
        }

        private static void LoadModel(KerasLayersJson layersJson, List<NNLayerBase> Layers)
        {
            foreach (var layer in layersJson.layers)
            {
                switch (layer.class_name)
                {
                    case "InputLayer":
                        var Input = new InputLayer();
                        Input.LoadConfig(layer.config);
                        Layers.Add(Input);
                        break;
                    case "Activation":
                        if (layer.config.activation == "relu")
                        {
                            Layers.Add(new ReLU());
                        }

                        if (layer.config.activation == "tanh")
                        {
                            Layers.Add(new Tanh());
                        }

                        break;
                    case "Conv2D":
                        var Conv2DLayer = new Conv2D();
                        Conv2DLayer.LoadConfig(layer.config);
                        Layers.Add(Conv2DLayer);
                        if (layer.config.activation == "relu")
                        {
                            Layers.Add(new ReLU());
                        }

                        if (layer.config.activation == "tanh")
                        {
                            Layers.Add(new Tanh());
                        }

                        break;
                    case "LeakyReLU":
                        var LeakyReLULayer = new LeakyReLU();
                        LeakyReLULayer.LoadConfig(layer.config);
                        Layers.Add(LeakyReLULayer);
                        break;
                    case "BatchNormalization":
                        Layers.Add(new BatchNormalization());
                        break;
                    case "UpSampling2D":
                        var UpSampling2DLayer = new UpSampling2D();
                        UpSampling2DLayer.LoadConfig(layer.config);
                        Layers.Add(UpSampling2DLayer);
                        break;
                    case "Concatenate":
                        var thislayer = new Concatenate();
                        string alternativeLayerName = layer.inbound_nodes[0][1][0] as string;
                        thislayer.AlternativeInputId =
                            Layers.FindIndex(ly => string.Compare(ly.Name, alternativeLayerName) == 0);
                        Layers.Add(thislayer);
                        break;
                    case "Add":
                        var addlayer = new Add();
                        int alterinput = layer.inbound_nodes[0].FindIndex(node =>
                            string.Compare(node[0] as string, Layers[Layers.Count - 1].Name) != 0);
                        string addalternativeLayerName = layer.inbound_nodes[0][alterinput][0] as string;
                        addlayer.AlternativeInputId =
                            Layers.FindIndex(ly => string.Compare(ly.Name, addalternativeLayerName) == 0);
                        Layers.Add(addlayer);
                        break;
                }

                Layers[Layers.Count - 1].Name = layer.name;
            }
        }

        private static void LoadWeight(List<KerasLayerWeightJson> weights, List<NNLayerBase> Layers)
        {
            int weightcount = 0;
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i] is Conv2D)
                {
                    (Layers[i] as Conv2D).LoadWeight(new KerasLayerWeightJson[2]
                    {
                        weights[weightcount],
                        weights[weightcount + 1]
                    });
                    weightcount += 2;
                }

                if (Layers[i] is BatchNormalization)
                {
                    (Layers[i] as BatchNormalization).LoadWeight(new KerasLayerWeightJson[4]
                    {
                        weights[weightcount],
                        weights[weightcount + 1],
                        weights[weightcount + 2],
                        weights[weightcount + 3]
                    });
                    weightcount += 4;
                }
            }
        }
    }
}

