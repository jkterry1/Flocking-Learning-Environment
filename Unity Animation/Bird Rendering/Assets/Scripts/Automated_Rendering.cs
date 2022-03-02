using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;
//using UnityEditor.Recorder;
//using UnityEditor.Recorder.Input;
using FFmpegOut;

public class Automated_Rendering : MonoBehaviour
{
    public String logs_folder;
    List<Controller> controllers;
    public GameObject camera;
    public GameObject birdPrefab;
    public int numBirds;
    public bool record;
    public bool in_editor;

    private IEnumerator coroutine;

    // Start is called before the first frame update
    void Start(){
      coroutine = Run();
      StartCoroutine(coroutine);
    }

    private void StartRecorder(String f, int s)
    {
        string[] path = f.Split('/');

        //calculate reward to determine what folder to put the video in
        int reward = (int)(float.Parse(path[path.Length-1].Split('_')[2]))/10;
        string folder = "";
        folder = reward.ToString()+"0s";
        Debug.Log("folder: "+folder);
        string fileName = path[path.Length-1].Substring(0, path[path.Length-1].Length-4)+"-render";
        fileName = f.Substring(0, f.Length-path[path.Length-1].Length-path[path.Length-2].Length-1) + folder + "-videos/" + fileName;

    }


    private IEnumerator Run()
    {
      DirectoryInfo dir = null;
      if(!in_editor){
        String[] path_split = Application.dataPath.Split('/');
        String path = "";
        for (int i=1; i<path_split.Length - 2; i++){
          path = path + "/" + path_split[i];
        }

        dir = new DirectoryInfo(path + @"/" + logs_folder);
      }
      else{
        dir = new DirectoryInfo(Application.dataPath + "/Scripts/"+ logs_folder);
      }
      FileInfo[] info = dir.GetFiles("*bird*.csv");

      List<String> files = new List<String>();

      foreach (FileInfo f in info)
      {
        files.Add(f.ToString());
      }

      CameraCapture camcap = camera.GetComponent<CameraCapture>();
      files = files.Distinct().ToList();
      foreach (String f in files)
      {
        //Debug.Log(f);
        Controller controller = gameObject.AddComponent(typeof(Controller)) as Controller;
        controller.numBirds = numBirds;
        controller.bird_filename = f;
        controller.camera = camera;
        controller.birdPrefab = birdPrefab;

        controller.Init_Animation();
        if(record){
          camcap.enabled = true;
        }
        //yield return new WaitForSeconds(4);
        controller.Start_Animation();
        int waitTime = controller.maxFrame/50 + 2;
        yield return new WaitForSeconds(waitTime);
        if(record){
          camcap.enabled = false;
        }
      }
      Application.Quit();

    }


}
