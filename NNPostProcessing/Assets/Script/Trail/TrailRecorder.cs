using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public class TrailRecorder : MonoBehaviour
{

    private CameraTrail tr;
	// Use this for initialization
	void Start ()
	{
	    tr = ScriptableObject.CreateInstance<CameraTrail>();
    }
	
	// Update is called once per frame
	void Update () {
	    tr.Positions.Add(transform.position);
	    tr.Rotations.Add(transform.rotation);
    }

    void OnDisable()
    {
        AssetDatabase.CreateAsset(tr, "Assets/Script/trail.asset");
        AssetDatabase.SaveAssets();
        Debug.Log("exit");
    }
}
