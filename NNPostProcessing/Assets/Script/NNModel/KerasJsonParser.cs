using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using Newtonsoft.Json;

namespace NNPP
{

    [System.Serializable]
    public class KerasJson
    {
        public KerasModelJson model;
        public List<KerasLayerWeightJson> weights;
    }

    [System.Serializable]
    public class KerasLayerWeightJson
    {
        public int[] shape;
        public float[] arrayweight;
        public float[,,,] kernelweight;
    }

    [System.Serializable]
    public class KerasModelJson
    {
        public string class_name;
        public KerasLayersJson config;
    }

    [System.Serializable]
    public class KerasLayersJson
    {
        public string name;
        public KerasLayerJson[] layers;
    }

    [System.Serializable]
    public class KerasLayerJson
    {
        public string name;
        public string class_name;
        public KerasLayerConfigJson config;
        public List<List<List<object>>> inbound_nodes;
    }

    [System.Serializable]
    public class KerasLayerConfigJson
    {
        public string name;
        public int filters;
        public int[] kernel_size;
        public int[] strides;
        public int[] size;
        public float alpha;
        public float momentum;
        public string activation;
    }
}