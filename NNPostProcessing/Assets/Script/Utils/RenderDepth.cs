using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RenderDepth : MonoBehaviour
{
    void Start()
    {
        gameObject.GetComponent<Camera>().depthTextureMode = DepthTextureMode.DepthNormals;
    }

    public Material mat;
    void OnRenderImage(RenderTexture src, RenderTexture dst)
    {
        Graphics.Blit(src, dst, mat);
    }
}
