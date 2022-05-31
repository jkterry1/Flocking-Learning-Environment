using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Bird{
    public int id;
    public List<State> states = new List<State>();
    public GameObject birdModel;
    public GameObject leftWing;
    public GameObject rightWing;
    public GameObject body;

    public Bird(int myID){
        id = myID;
        states = new List<State>();
    }

    void Start(){
      states = new List<State>();
    }

    public void AddModel(GameObject model){
        birdModel = model;
        body = birdModel.transform.Find("parts").gameObject;
        leftWing = birdModel.transform.Find("parts").Find("left wing").gameObject;
        rightWing = birdModel.transform.Find("parts").Find("right wing").gameObject;
    }

    public void Step(int curFrame){

        Vector3 pos = new Vector3(states[curFrame].x,
                                  states[curFrame].y,
                                  states[curFrame].z);
        //birdModel.transform.position = pos;
        body.transform.position = pos;

        Vector3 orient = new Vector3(states[curFrame].pitch,
                                states[curFrame].yaw,
                                states[curFrame].roll);
        //birdModel.transform.eulerAngles = orient;
        body.transform.eulerAngles = orient;

        Vector3 leftWingOrientation = new Vector3(states[curFrame].betaLeft, 0.0f,states[curFrame].alphaLeft);
        leftWing.transform.eulerAngles = orient+leftWingOrientation;
        //leftWing.transform.eulerAngles = leftWingOrientation;

        Vector3 rightWingOrientation = new Vector3(-states[curFrame].betaRight, 0.0f, -states[curFrame].alphaRight);
        rightWing.transform.eulerAngles = orient-rightWingOrientation;
        //rightWing.transform.eulerAngles = rightWingOrientation;
    }

    public void AddState(State state){
        if(states == null){
          states = new List<State>();
        }
        states.Add(state);
    }
}
