using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;
using UnityEditor.Recorder;
using UnityEditor.Recorder.Input;

public class Automated_Rendering : MonoBehaviour
{
    public String logs_folder;
    List<Controller> controllers;
    public GameObject camera;
    public GameObject birdPrefab;
    public int numBirds;
    public bool record;

    private IEnumerator coroutine;

    private RecorderController TestRecorderController;


    // Start is called before the first frame update
    void Start(){
      coroutine = Run();
      StartCoroutine(coroutine);
    }

    private void StartRecorder(String f, int s)
    {
        var controllerSettings = ScriptableObject.CreateInstance<RecorderControllerSettings>();
        TestRecorderController = new RecorderController(controllerSettings);
        var videoRecorder = ScriptableObject.CreateInstance<MovieRecorderSettings>();
        videoRecorder.name = "My Video Recorder";
        videoRecorder.Enabled = true;
        videoRecorder.VideoBitRateMode = VideoBitrateMode.High;
        videoRecorder.ImageInputSettings = new GameViewInputSettings
        {
            OutputWidth = 640,
            OutputHeight = 480
        };
        videoRecorder.AudioInputSettings.PreserveAudio = true;
        string[] path = f.Split('/');

        //calculate reward to determine what folder to put the video in
        int reward = (int)(float.Parse(path[path.Length-1].Split('_')[2]))/10;
        string folder = "";
        folder = reward.ToString()+"0s";
        Debug.Log("folder: "+folder);
        string fileName = path[path.Length-1].Substring(0, path[path.Length-1].Length-4)+"-render";
        fileName = f.Substring(0, f.Length-path[path.Length-1].Length-path[path.Length-2].Length-1) + folder + "-videos/" + fileName;
        videoRecorder.OutputFile = fileName;
        controllerSettings.AddRecorderSettings(videoRecorder);
        controllerSettings.SetRecordModeToFrameInterval(0, s*10 + 5); // s seconds @ 10 FPS
        controllerSettings.FrameRate = 10;
        RecorderOptions.VerboseMode = false;
        TestRecorderController.PrepareRecording();
        TestRecorderController.StartRecording();
    }

    private void StopRecorder()
    {
        TestRecorderController.StopRecording();
    }

    private IEnumerator Run()
    {
      DirectoryInfo dir = new DirectoryInfo(Application.dataPath + @"/Scripts/" + logs_folder);
      FileInfo[] info = dir.GetFiles("*bird*.csv");

      List<String> files = new List<String>();

      foreach (FileInfo f in info)
      {
        files.Add(f.ToString());
      }

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
          StartRecorder(f, controller.maxFrame/50);
        }
        //yield return new WaitForSeconds(4);
        controller.Start_Animation();
        int waitTime = controller.maxFrame/50 + 2;
        yield return new WaitForSeconds(waitTime);
      }
      Application.Quit();

    }

    // Update is called once per frame
    void Update()
    {

    }

}
