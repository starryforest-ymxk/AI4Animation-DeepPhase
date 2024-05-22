#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using System.Linq;
using AI4Animation;

namespace DeepPhase
{
    public class DancePipeline : AssetPipelineSetup 
    {
        public enum Mode {ProcessAssets, ExportController};
        public Mode mode = Mode.ProcessAssets;
        public string audioPath = string.Empty;
        
        public int channels = 8;
        public bool writeMirror = true;
        public bool updateAudios = true;
        public bool updateContact = true;

        private DateTime m_timestamp;
        private float m_progress;
        private float m_samplesPerSecond;
        private int m_samples;
        private int m_sequence;
        private string m_audioPath = String.Empty;
        private bool m_updateAudioAssets = true;
        private bool m_updateContact = true;
        
        private TimeSeries m_musicSerious;
        private int m_pastKeys = 20;
        private int m_futureKeys = 20;
        private float m_pastWindow = 1f;
        private float m_futureWindow = 1f;
        private int m_resolution = 1;

        private AssetPipeline.Data.File s;
        private AssetPipeline.Data x, y;
        
        public override void Inspector() {
            
           mode = (Mode)EditorGUILayout.EnumPopup("Mode", mode);

           if(mode == Mode.ProcessAssets)
           {
               ProcessAssetsMode();
           }
           else if(mode == Mode.ExportController) 
           {
               ExportMode();
           }
           
           void ExportMode() {
               EditorGUI.BeginDisabledGroup(true);
               EditorGUILayout.FloatField("Export Framerate", Pipeline.GetEditor().TargetFramerate);
               EditorGUILayout.TextField("Export Path", AssetPipeline.Data.GetExportPath());
               EditorGUI.EndDisabledGroup(); 
               channels = EditorGUILayout.IntField("Channels", channels);
               writeMirror = EditorGUILayout.Toggle("Write Mirror", writeMirror);
               Utility.SetGUIColor(UltiDraw.DarkGrey);
               using(new EditorGUILayout.VerticalScope ("Box")) {
                   Utility.ResetGUIColor();
                   EditorGUILayout.LabelField("Music Series");
                   m_pastKeys = Mathf.Max(EditorGUILayout.IntField("Past Keys", m_pastKeys), 0);
                   m_futureKeys = Mathf.Max(EditorGUILayout.IntField("Future Keys", m_futureKeys), 0);
                   m_pastWindow = EditorGUILayout.FloatField("Past Window", m_pastWindow);
                   m_futureWindow = EditorGUILayout.FloatField("Future Window", m_futureWindow);
                   m_resolution = Mathf.Max(EditorGUILayout.IntField("Resolution", m_resolution), 1);
               }
               if(Pipeline.IsProcessing() || Pipeline.IsAborting()) {
                   EditorGUI.BeginDisabledGroup(true);
                   EditorGUILayout.FloatField("Samples Per Second", m_samplesPerSecond);
                   EditorGUI.EndDisabledGroup();
                   EditorGUI.DrawRect(
                       new Rect(
                           EditorGUILayout.GetControlRect().x, 
                           EditorGUILayout.GetControlRect().y, 
                           m_progress * EditorGUILayout.GetControlRect().width, 20f
                       ), 
                       UltiDraw.Green.Opacity(0.5f)
                   );
               }
           }

           void ProcessAssetsMode()
           {
               EditorGUILayout.BeginHorizontal();
               EditorGUILayout.LabelField("Update Contact Sensors", GUILayout.Width(150));
               updateContact = EditorGUILayout.Toggle(updateContact);
               EditorGUILayout.EndHorizontal();
               
               EditorGUILayout.BeginHorizontal();
               EditorGUILayout.LabelField("Update Audio Assets", GUILayout.Width(150));
               updateAudios = EditorGUILayout.Toggle(updateAudios);
               EditorGUILayout.EndHorizontal();
               if (updateAudios)
               {
                   EditorGUILayout.LabelField("Audio Path");
               
                   EditorGUILayout.BeginHorizontal();
                   EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
                   audioPath = EditorGUILayout.TextField(audioPath);
                   EditorGUILayout.EndHorizontal();
               }
               
               if(updateContact && (Pipeline.IsProcessing() || Pipeline.IsAborting())) {
                   EditorGUILayout.LabelField("Capture Contact");
                   EditorGUI.DrawRect(
                       new Rect(
                           EditorGUILayout.GetControlRect().x, 
                           EditorGUILayout.GetControlRect().y, 
                           m_progress * EditorGUILayout.GetControlRect().width, 20f
                       ), 
                       UltiDraw.Green.Opacity(0.5f)
                   );
               }
           }
        }

        public override void Inspector(BatchProcessor.Item item) { }

        public override bool CanProcess() { return true; }

        public override void Begin()
        {
            if(mode == Mode.ProcessAssets) {
                m_audioPath = string.IsNullOrEmpty(audioPath)?"Assets":"Assets/" + audioPath;
                m_progress = 0;
                m_updateAudioAssets = updateAudios;
                m_updateContact = updateContact;
            }
            if(mode == Mode.ExportController) {
                m_samples = 0;
                m_sequence = 0;
                m_progress = 0;
                m_musicSerious = new TimeSeries(m_pastKeys, m_futureKeys, m_pastWindow, m_futureWindow, m_resolution);
                s = AssetPipeline.Data.CreateFile("Sequences", AssetPipeline.Data.TYPE.Text);
                x = new AssetPipeline.Data("Input");
                y = new AssetPipeline.Data("Output");
            }
        }

        public override IEnumerator Iterate(MotionAsset asset)
        {
            Pipeline.GetEditor().LoadSession(Utility.GetAssetGUID(asset));
            
            if(mode == Mode.ProcessAssets) {
                yield return EditorCoroutines.StartCoroutine(ProcessAssets(asset), this);
            }
            
            if(mode == Mode.ExportController) {
                yield return EditorCoroutines.StartCoroutine(ProcessControllerAssets(asset), this);
            }

            yield return new WaitForSeconds(0);
        }
        
        public override void Callback()
        {
            if(mode == Mode.ProcessAssets) {
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
                Resources.UnloadUnusedAssets();
            }
            if(mode == Mode.ExportController) {
                Resources.UnloadUnusedAssets();
            }
        }

        public override void Finish()
        {
            if(mode == Mode.ProcessAssets) {
                foreach (var t in Pipeline.GetEditor().Assets) {
                    //MotionAsset.Retrieve(Pipeline.GetEditor().Assets[i]).ResetSequences();
                    MotionAsset.Retrieve(t).Export = true;
                }
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[0]).SetSequence(0, 2000, 48000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[1]).SetSequence(0, 3800, 23000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[2]).SetSequence(0, 2500, 26500);
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[3]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[4]).Export = false;
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[5]).SetSequence(0, 700, 43000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[6]).SetSequence(0, 500, 20000);
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[7]).SetSequence(0, 2000, 23000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[8]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[9]).SetSequence(0, 2000, 24000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[7]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[8]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[9]).Export = false;
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[10]).SetSequence(0, 2000, 30000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[11]).SetSequence(0, 1000, 16000);
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[12]).SetSequence(0, 1000, 19500);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[13]).SetSequence(0, 1000, 19500);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[14]).SetSequence(0, 2000, 30000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[12]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[13]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[14]).Export = false;
                
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[15]).SetSequence(0, 2000, 30000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[16]).SetSequence(0, 1500, 24000);
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[17]).SetSequence(0, 1000, 8000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[18]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[17]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[18]).Export = false;
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[19]).SetSequence(0, 1500, 20000);
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[20]).SetSequence(0, 1000, 18000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[21]).SetSequence(0, 2000, 18000);
                
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[22]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[23]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[24]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[25]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[26]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[27]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[22]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[23]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[24]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[25]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[26]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[27]).Export = false;
                
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[28]).SetSequence(0, 2000, 30000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[29]).SetSequence(0, 2000, 30000);
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[30]).SetSequence(0, 2000, 30000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[31]).SetSequence(0, 2000, 40000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[32]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[33]).SetSequence(0, 2000, 30000);
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[34]).SetSequence(0, 2000, 10000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[35]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[36]).SetSequence(0, 2000, 3000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[37]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[34]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[35]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[36]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[37]).Export = false;
                
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[38]).SetSequence(0, 2000, 10000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[39]).SetSequence(0, 2000, 25000);
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[40]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[41]).SetSequence(0, 2000, 15000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[42]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[43]).SetSequence(0, 2000, 15000);
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[44]).SetSequence(0, 2000, 30000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[45]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[46]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[47]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[48]).SetSequence(0, 2000, 18000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[49]).SetSequence(0, 2000, 10000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[44]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[45]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[46]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[47]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[48]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[49]).Export = false;
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[50]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[51]).SetSequence(0, 2000, 20000);
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[52]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[53]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[54]).SetSequence(0, 2000, 15000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[55]).SetSequence(0, 2000, 20000);
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[56]).SetSequence(0, 2000, 15000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[52]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[53]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[54]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[55]).Export = false;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[56]).Export = false;
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[57]).SetSequence(0, 2000, 10000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[58]).SetSequence(0, 2000, 15000);
                
                // MotionAsset.Retrieve(Pipeline.GetEditor().Assets[59]).SetSequence(0, 2000, 20000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[59]).Export = false;
                
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[60]).SetSequence(0, 2000, 15000);
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[61]).SetSequence(0, 2000, 20000);
                
                foreach (var t in Pipeline.GetEditor().Assets) {
                    MotionAsset.Retrieve(t).MarkDirty(true, false);
                }
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
                Resources.UnloadUnusedAssets();
            }
            if(mode == Mode.ExportController) {                
                s.Close();
                x.Finish();
                y.Finish();
            }
        }
        
        private IEnumerator ProcessAssets(MotionAsset asset) {
            
            asset.MirrorAxis = Axis.XPositive;
            asset.Model = "Character";
            asset.Scale = 0.01f;
            asset.Export = true;
            asset.ClearSequences();
            
            {
                RootModule module = asset.HasModule<RootModule>() ? asset.GetModule<RootModule>() : asset.AddModule<RootModule>();
                module.Topology = RootModule.TOPOLOGY.Biped;
                module.SmoothRotations = true;
            }
            
            {
                ContactModule module = asset.HasModule<ContactModule>() ? asset.GetModule<ContactModule>() : asset.AddModule<ContactModule>();
                if (m_updateContact)
                {
                    module.Clear();
                    module.AddSensor("LeftFoot", Vector3.zero, Vector3.zero, 1f/7f*Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                    module.AddSensor("RightFoot", Vector3.zero, Vector3.zero, 1f/7f*Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                    yield return EditorCoroutines.StartCoroutine(module.CaptureContactsAsync(Pipeline.GetEditor(), t=>m_progress = t), this);
                }
            }
            
            {
                AudioSpectrumModule module = asset.HasModule<AudioSpectrumModule>() ? asset.GetModule<AudioSpectrumModule>() : asset.AddModule<AudioSpectrumModule>();
                if (m_updateAudioAssets)
                {
                    string audioName = asset.name.EndsWith(".bvh") ? asset.name.Substring(0, asset.name.Length - 4) : asset.name;
                    string searchFilter = $"{audioName} t:AudioSpectrum";
                    string[] guids = AssetDatabase.FindAssets(searchFilter, new[] { m_audioPath });
                    AudioSpectrum[] audios = new AudioSpectrum[guids.Length];
                    for(int i = 0; i < guids.Length; i++)
                    {
                        string path = AssetDatabase.GUIDToAssetPath(guids[i]);
                        audios[i] = AssetDatabase.LoadAssetAtPath<AudioSpectrum>(path);
                    }
                
                    if (audios.Length == 1 && audios[0].Clip != null)
                    {
                        module.AudioSpectrum = audios[0];
                        module.AudioSpectrums = Array.Empty<AudioSpectrum>();
                    }
                    else if (audios.Length > 1 && audios.All(audio => audio.Clip != null))
                    {
                        module.AudioSpectrum = null;
                        module.AudioSpectrums = audios;
                    }
                    else
                    {
                        module.AudioSpectrum = null;
                        module.AudioSpectrums = Array.Empty<AudioSpectrum>();
                        Debug.LogError($"AudioSpectrum called {audioName} was not set properly in {m_audioPath}.");
                        yield break;
                    }
                }
            }
            
            asset.MarkDirty(true, false);
            yield return new WaitForSeconds(0);
        }

        private IEnumerator ProcessControllerAssets(MotionAsset asset)
        {
            if(asset.Export) {
                for(int i=1; i<=2; i++) {
                    if(i==1) {
                        Pipeline.GetEditor().SetMirror(false);
                    } else if(i==2 && writeMirror) {
                        Pipeline.GetEditor().SetMirror(true);
                    } else {
                        break;
                    }
                    foreach(Interval seq in asset.Sequences) {
                        m_sequence += 1;
                        float start = asset.GetFrame(asset.GetFrame(seq.Start).Timestamp).Timestamp;
                        float end = asset.GetFrame(asset.GetFrame(seq.End).Timestamp - 1f/Pipeline.GetEditor().TargetFramerate).Timestamp;
                        int index = 0;
                        while(Pipeline.IsProcessing() && (start + index/Pipeline.GetEditor().TargetFramerate < end || Mathf.Approximately(start + index/Pipeline.GetEditor().TargetFramerate, end))) {
                            float tCurrent = start + index/Pipeline.GetEditor().TargetFramerate;
                            float tNext = start + (index+1)/Pipeline.GetEditor().TargetFramerate;
                            index +=1;
        
                            DanceControllerSetup.Export(this, x, y, tCurrent, tNext, m_musicSerious);
        
                            x.Store();
                            y.Store();
                            WriteSequenceInfo(m_sequence, tCurrent, Pipeline.GetEditor().Mirror, asset);
                            
                            m_samples += 1;
                            if(Utility.GetElapsedTime(m_timestamp) >= 0.2f) {
                                m_progress = (Math.Abs(end - start) <0.001f) ? 1f : ((tCurrent-start) / (end-start));
                                m_samplesPerSecond = m_samples / (float)Utility.GetElapsedTime(m_timestamp);
                                m_samples = 0;
                                m_timestamp = Utility.GetTimestamp();
                                yield return new WaitForSeconds(0);
                            }
                        }
                    }
                }
                yield return new WaitForSeconds(0);
            }

        }
        
        private void WriteSequenceInfo(int sequence, float timestamp, bool mirrored, MotionAsset asset) {
            //Sequence - Timestamp - Mirroring - Name - GUID
            s.WriteLine(
                sequence + AssetPipeline.Data.Separator + 
                timestamp + AssetPipeline.Data.Separator +
                (mirrored ? "Mirrored" : "Standard") + AssetPipeline.Data.Separator +
                asset.name + AssetPipeline.Data.Separator +
                Utility.GetAssetGUID(asset));
        }

        private static class DanceControllerSetup {
            public static void Export(DancePipeline setup, AssetPipeline.Data x, AssetPipeline.Data y, float tCurrent, float tNext, TimeSeries musicSeries) {
                Container current = new Container(setup, tCurrent, musicSeries);
                Container next = new Container(setup, tNext, musicSeries);
                
                string[] contacts = {"LeftFoot", "RightFoot"};
    
                //Input
                //Control
                for(int k=0; k<current.TimeSeries.Samples.Length; k++) {
                    x.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(current.Root), "TrajectoryPosition"+(k+1));
                    x.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(current.Root), "TrajectoryDirection"+(k+1));
                    x.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(current.Root), "TrajectoryVelocity"+(k+1));
                }
                
                //Audio Features
                for(int k=0; k<current.SpectrumSeries.Samples.Length; k++)
                {
                    x.Feed(current.SpectrumSeries.Values[k].Spectogram, "Spectogram" + (k+1) + "-");
                    x.Feed(current.SpectrumSeries.Values[k].Beats, "Beats" + (k+1) + "-");
                    x.Feed(current.SpectrumSeries.Values[k].Flux, "Flux" + (k+1) + "-");
                    x.Feed(current.SpectrumSeries.Values[k].MFCC, "MFCC" + (k+1) + "-");
                    x.Feed(current.SpectrumSeries.Values[k].Chroma, "Chroma" + (k+1) + "-");
                    x.Feed(current.SpectrumSeries.Values[k].ZeroCrossing, "ZeroCrossing" + (k+1) + "-");
                }

                //Auto-Regressive Posture
                for(int k=0; k<current.ActorPosture.Length; k++) {
                    x.Feed(current.ActorPosture[k].GetPosition().PositionTo(current.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Position");
                    x.Feed(current.ActorPosture[k].GetForward().DirectionTo(current.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Forward");
                    x.Feed(current.ActorPosture[k].GetUp().DirectionTo(current.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Up");
                    x.Feed(current.ActorVelocities[k].DirectionTo(current.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Velocity");
                }

                //Gating Variables
                x.Feed(current.PhaseSeries.GetAlignment(), "PhaseSpace-");

                //Output
                //Root Update
                Matrix4x4 delta = next.Root.TransformationTo(current.Root);
                y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
                y.FeedXZ(next.RootSeries.Velocities[next.TimeSeries.Pivot].DirectionTo(next.Root), "RootVelocity");
               

                //Control
                for(int k=next.TimeSeries.Pivot+1; k<next.TimeSeries.Samples.Length; k++) {
                    y.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(next.Root), "TrajectoryPosition"+(k+1));
                    y.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(next.Root), "TrajectoryDirection"+(k+1));
                    y.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(next.Root), "TrajectoryVelocity"+(k+1));
                }

                //Auto-Regressive Posture
                for(int k=0; k<next.ActorPosture.Length; k++) {
                    y.Feed(next.ActorPosture[k].GetPosition().PositionTo(next.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Position");
                    y.Feed(next.ActorPosture[k].GetForward().DirectionTo(next.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Forward");
                    y.Feed(next.ActorPosture[k].GetUp().DirectionTo(next.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Up");
                    y.Feed(next.ActorVelocities[k].DirectionTo(next.Root), "Bone"+(k+1)+setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName()+"Velocity");
                }

                //Contacts
                y.Feed(next.ContactSeries.GetContacts(next.TimeSeries.Pivot, contacts), "Contacts-");

                //Phase Update
                y.Feed(next.PhaseSeries.GetUpdate(), "PhaseUpdate-");
            }

            private class Container {
                public MotionAsset Asset;
                public Frame Frame;
                public Actor Actor;

                public TimeSeries TimeSeries;
                
                public RootModule.Series RootSeries;
                public ContactModule.Series ContactSeries;
                public DeepPhaseModule.Series PhaseSeries;
                public AudioSpectrumModule.Series SpectrumSeries;

                //Actor Features
                public Matrix4x4 Root;
                public Matrix4x4[] ActorPosture;
                public Vector3[] ActorVelocities;

                public Container(DancePipeline setup, float timestamp, TimeSeries musicSeries) {
                    MotionEditor editor = setup.Pipeline.GetEditor();
                    editor.LoadFrame(timestamp);
                    Asset = editor.GetSession().Asset;
                    Frame = editor.GetCurrentFrame();

                    TimeSeries = editor.GetTimeSeries();
                    RootSeries = Asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as RootModule.Series;
                    ContactSeries = Asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as ContactModule.Series;
                    PhaseSeries = Asset.GetModule<DeepPhaseModule>(setup.channels+"Channels").ExtractSeries(TimeSeries, timestamp, editor.Mirror) as DeepPhaseModule.Series;
                    SpectrumSeries = Asset.GetModule<AudioSpectrumModule>().ExtractSeries(musicSeries, timestamp, editor.Mirror) as AudioSpectrumModule.Series;
                    
                    Root = editor.GetSession().GetActor().transform.GetWorldMatrix();
                    ActorPosture = editor.GetSession().GetActor().GetBoneTransformations();
                    ActorVelocities = editor.GetSession().GetActor().GetBoneVelocities();
                }
            }
        }
    }
}


#endif
