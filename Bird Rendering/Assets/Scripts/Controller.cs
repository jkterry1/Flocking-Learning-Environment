using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Controller : MonoBehaviour{
    public string bird_filename = "bird_log_1.csv";
    public string vortex_filename = "vortex_log_1.csv";
    public int numBirds;

    public List<Vector3> path;
    public List<Bird> birds;
    List<Vortex> vortices = new List<Vortex>();
    public int curFrame;
    public int maxFrame;
    public GameObject birdPrefab;
    public GameObject vortexPrefab;
    public GameObject camera;

    int index;
    public bool render_vortices = false;
    float time = 0.0f;
    List<Vortex> visible_vortices = new List<Vortex>();
    int i=0;

    public bool start;
    public bool running = false;

    //void Start(){}

    public void Init_Animation(){
        running = true;
        curFrame = 0;
        maxFrame = 0;
        //string[] bird_lines = System.IO.File.ReadAllLines(Application.dataPath + @"/Scripts/" + bird_filename);
        string[] bird_lines = System.IO.File.ReadAllLines(bird_filename);
        //Debug.Log(bird_filename);
        birds = new List<Bird>();

        for (int i = 0; i < numBirds; i++){
            //Bird temp = gameObject.AddComponent(typeof(Bird)) as Bird;
            //temp.id = i;
            Bird temp = new Bird(i);
            birds.Add(temp);
            //Debug.Log(birds.Count());
            GameObject obj = Instantiate(birdPrefab, new Vector3(0, 0, 0), Quaternion.identity);
            birds[birds.Count() - 1].AddModel(obj);
        }

        index = birds.Count()/2;
        MoveCamera();

        //int lines = 0;

        foreach (string line in bird_lines){
            //lines++;
            string[] elements = line.Split(',');

            int id = Int32.Parse(elements[0]);

            float x = float.Parse(elements[3]);
            float y = float.Parse(elements[4]);
            float z = float.Parse(elements[2]);

            float roll = float.Parse(elements[5]) * Mathf.Rad2Deg;
            float pitch = float.Parse(elements[6]) * Mathf.Rad2Deg;
            float yaw = float.Parse(elements[7]) * Mathf.Rad2Deg + 90.0f;

            float alphaLeft = float.Parse(elements[8]) * Mathf.Rad2Deg;
            float alphaRight = float.Parse(elements[9]) * Mathf.Rad2Deg;
            float betaLeft = float.Parse(elements[10]) * Mathf.Rad2Deg;
            float betaRight = float.Parse(elements[11]) * Mathf.Rad2Deg;

            birds[id].AddState(new State(x, y, z, roll, pitch, yaw, alphaLeft, alphaRight, betaLeft, betaRight));
            //Debug.Log(birds[id].states.Count());
        }

        foreach (Bird bird in birds){
            if (bird.states.Count() >= maxFrame){
                maxFrame = bird.states.Count();
            }
        }
    }

    public void Start_Animation(){
        InvokeRepeating("StepBirds", 0.05f, 0.02f);
        //InvokeRepeating("StepVortices", 2.0f, 0.1f);
    }

    void StepBirds()
    {
      foreach (Bird bird in birds){
          bird.Step(curFrame);
      }

      curFrame++;
      //curFrame = curFrame % maxFrame;
      // Cancel all Invoke calls
      if (curFrame >= maxFrame){
        running = false;
        camera.transform.parent = null;
        RemoveBirds();
        CancelInvoke();
      }
    }

    void RemoveBirds(){
      foreach(Bird bird in birds){
        Destroy(bird.birdModel);
      }
    }

    void StepVortices(){
      List<Vortex> temp = new List<Vortex>();
      while(vortices[i].time <= time){
        temp.Add(vortices[i]);
        vortices[i].spawn();
        i++;
      }
      foreach (Vortex v in visible_vortices){
        v.remove();
        Destroy(v);
      }
      visible_vortices = temp;
      time += 0.1f;
    }


    // Update is called once per frame
    void Update(){
      if(running){
        MoveCamera();
      }
    }

    void MoveCamera(){
      camera.transform.position = new Vector3(birds[index].body.transform.position.x - 0.0f,
                                              birds[index].body.transform.position.y + 60.0f,
                                              birds[index].body.transform.position.z - 60.0f);
      camera.transform.LookAt(birds[index].body.transform);
    }
}
