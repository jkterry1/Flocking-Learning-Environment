
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
    float period = 0.0f;
    public bool render_vortices = false;
    float time = 0.0f;
    List<Vortex> visible_vortices = new List<Vortex>();
    int i=0;

    // Start is called before the first frame update
    void Start(){
        curFrame = 0;
        maxFrame = 0;
        string[] bird_lines = System.IO.File.ReadAllLines(Application.dataPath + @"/Scripts/" + bird_filename);
        birds = new List<Bird>();

        for (int i = 0; i < numBirds; i++){
            birds.Add(new Bird(i));
            GameObject obj = Instantiate(birdPrefab, new Vector3(0, 0, 0), Quaternion.identity);
            birds[birds.Count() - 1].AddModel(obj);
        }

        foreach (string line in bird_lines){
            string[] elements = line.Split(',');

            int id = Int32.Parse(elements[0]);

            float x = float.Parse(elements[3]);
            float y = float.Parse(elements[4]);
            float z = float.Parse(elements[2]);

            float roll = float.Parse(elements[5]) * Mathf.Rad2Deg;
            float pitch = float.Parse(elements[6]) * Mathf.Rad2Deg;
            float yaw = float.Parse(elements[7]) * Mathf.Rad2Deg;

            float alphaLeft = float.Parse(elements[8]) * Mathf.Rad2Deg;
            float alphaRight = float.Parse(elements[9]) * Mathf.Rad2Deg;
            float betaLeft = float.Parse(elements[10]) * Mathf.Rad2Deg;
            float betaRight = float.Parse(elements[11]) * Mathf.Rad2Deg;

            birds[id].AddState(new State(x, y, z, roll, pitch, yaw, alphaLeft, alphaRight, betaLeft, betaRight));
        }
        foreach (Bird bird in birds){
            if (bird.states.Count() >= maxFrame){
                maxFrame = bird.states.Count();
            }
        }

        string[] vortex_lines = System.IO.File.ReadAllLines(Application.dataPath + @"/Scripts/" + vortex_filename);
        foreach (string line in vortex_lines){
            string[] elements = line.Split(',');

            float time = float.Parse(elements[0]);

            float x = float.Parse(elements[2]);
            float y = float.Parse(elements[3]);
            float z = float.Parse(elements[1]);

            float roll = float.Parse(elements[4]) * Mathf.Rad2Deg;
            float pitch = float.Parse(elements[5]) * Mathf.Rad2Deg;
            float yaw = float.Parse(elements[6]) * Mathf.Rad2Deg;

            float gamma = float.Parse(elements[7]);

            vortices.Add(new Vortex(vortexPrefab, time, x, y, z, gamma));
        }

        foreach (Bird bird in birds){
            if (bird.states.Count() >= maxFrame){
                maxFrame = bird.states.Count();
            }
        }

        InvokeRepeating("StepBirds", 2.0f, 0.01f);
        InvokeRepeating("StepVortices", 2.0f, 0.1f);
    }

    void StepBirds()
    {
      foreach (Bird bird in birds){
          bird.Step(10*curFrame);
      }

      curFrame++;
      curFrame = curFrame % maxFrame;
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
      camera.transform.LookAt(birds[0].birdModel.transform);
    }
}
