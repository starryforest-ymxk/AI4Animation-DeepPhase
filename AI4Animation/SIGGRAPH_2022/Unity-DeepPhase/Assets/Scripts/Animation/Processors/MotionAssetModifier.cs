#if UNITY_EDITOR
using System;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Serialization;

namespace AI4Animation {
    public class MotionAssetModifier : BatchProcessor {

        public string Source = string.Empty;
        public MotionAsset standard = null;
        
        public float Framerate = 1f;
        public string Model = string.Empty;
        public Vector3 Translation = Vector3.zero;
        public Vector3 Rotation = Vector3.zero;
        public float Scale = 1f;
        public Axis MirrorAxis = Axis.XPositive;
        public bool Export = true;

        public bool modifyFramerate = false;
        public bool modifyModel = false;
        public bool modifyTranslation = false;
        public bool modifyRotation = false;
        public bool modifyScale = false;
        public bool modifyMirrorAxis = false;
        public bool modifyExport = false;
        public bool setAlignment = true;

        public bool modifyFramerateResult = false;
        public bool modifyModelResult = false;
        public bool modifyTranslationResult = false;
        public bool modifyRotationResult = false;
        public bool modifyScaleResult = false;
        public bool modifyMirrorAxisResult = false;
        public bool modifyExportResult = false;
        public bool setAlignmentResult = true;

		private List<string> Imported;
		private List<string> Skipped;

        [MenuItem ("AI4Animation/Tools/Motion Asset Modifier")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(MotionAssetModifier));
            Scroll = Vector3.zero;
        }

		public override string GetID(Item item) {
			return item.ID;
		}

        public override void DerivedRefresh() {
            
        }
        
        public override void DerivedInspector() {
			EditorGUILayout.LabelField("Source");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			SetSource(EditorGUILayout.TextField(Source));
			EditorGUILayout.EndHorizontal();

			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					
					EditorGUILayout.BeginHorizontal();
					modifyFramerate = EditorGUILayout.Toggle("Framerate", modifyFramerate, GUILayout.Width(200));
					if(modifyFramerate)
						Framerate = EditorGUILayout.FloatField(Framerate);
					EditorGUILayout.EndHorizontal();
					
					EditorGUILayout.BeginHorizontal();
					modifyModel = EditorGUILayout.Toggle("Model", modifyModel, GUILayout.Width(200));
					if(modifyModel)
						Model = EditorGUILayout.TextField(Model);
					EditorGUILayout.EndHorizontal();
			
					EditorGUILayout.BeginHorizontal();
					modifyTranslation = EditorGUILayout.Toggle("Translation", modifyTranslation, GUILayout.Width(200));
					if(modifyTranslation)
						Translation = EditorGUILayout.Vector3Field("", Translation);
					EditorGUILayout.EndHorizontal();
			
					EditorGUILayout.BeginHorizontal();
					modifyRotation = EditorGUILayout.Toggle("Rotation", modifyRotation, GUILayout.Width(200));
					if(modifyRotation)
						Rotation = EditorGUILayout.Vector3Field("", Rotation);
					EditorGUILayout.EndHorizontal();
			
					EditorGUILayout.BeginHorizontal();
					modifyScale = EditorGUILayout.Toggle("Scale", modifyScale, GUILayout.Width(200));
					if(modifyScale)
						Scale = EditorGUILayout.FloatField(Scale);
					EditorGUILayout.EndHorizontal();
			
					EditorGUILayout.BeginHorizontal();
					modifyMirrorAxis = EditorGUILayout.Toggle("MirrorAxis", modifyMirrorAxis, GUILayout.Width(200));
					if(modifyMirrorAxis)
						MirrorAxis = (Axis)EditorGUILayout.EnumPopup(MirrorAxis);
					EditorGUILayout.EndHorizontal();
					
					EditorGUILayout.BeginHorizontal();
					modifyExport = EditorGUILayout.Toggle("Export", modifyExport, GUILayout.Width(200));
					if(modifyExport)
						if(Utility.GUIButton("Export", Export ? UltiDraw.DarkGreen : UltiDraw.DarkRed, UltiDraw.White)) {
							Export = !Export;
						}
					EditorGUILayout.EndHorizontal();
					
					setAlignment = EditorGUILayout.Toggle("Set Alignment", setAlignment);
					
					if (setAlignment)
					{
						Utility.SetGUIColor(UltiDraw.White);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							
							EditorGUILayout.BeginHorizontal();
							standard = (MotionAsset)EditorGUILayout.ObjectField("Standard Motion Asset", standard, typeof(MotionAsset),standard);

							EditorGUILayout.EndHorizontal();

							if (standard != null)
							{
								EditorGUI.BeginChangeCheck();
								
								if(Utility.GUIButton("Detect Symmetry", UltiDraw.DarkGrey, UltiDraw.White)) {
									standard.DetectSymmetry();
								}
								for(int i=0; i<standard.Source.Bones.Length; i++) {
									EditorGUILayout.BeginHorizontal();
									EditorGUI.BeginDisabledGroup(true);
									EditorGUILayout.TextField(standard.Source.GetBoneNames()[i]);
									EditorGUI.EndDisabledGroup();
									standard.SetSymmetry(i, EditorGUILayout.Popup(standard.Symmetry[i], standard.Source.GetBoneNames()));
									standard.Source.Bones[i].Parent = EditorGUILayout.Popup(standard.Source.Bones[i].Parent+1, ArrayExtensions.Concat(new string[]{"None"}, standard.Source.GetBoneNames())) - 1; 
									EditorGUILayout.LabelField("Alignment", GUILayout.Width(60f));
									standard.Source.Bones[i].Alignment = EditorGUILayout.Vector3Field("", standard.Source.Bones[i].Alignment, GUILayout.Width(200f));
									EditorGUILayout.LabelField("Correction", GUILayout.Width(60f));
									standard.Source.Bones[i].Correction = EditorGUILayout.Vector3Field("", standard.Source.Bones[i].Correction, GUILayout.Width(200f));
									EditorGUILayout.LabelField("Override", GUILayout.Width(60f));
									standard.Source.Bones[i].Override = EditorGUILayout.TextField(standard.Source.Bones[i].Override, GUILayout.Width(100f));
									EditorGUILayout.EndHorizontal();
								}
								
								if (EditorGUI.EndChangeCheck())
								{
									standard.MarkDirty();
									AssetDatabase.SaveAssets();
								}
							}
						}
					}
				}
			}
			
			if(Utility.GUIButton("Load Source Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
				LoadDirectory(Source);
			}
        }

		private void SetSource(string source) {
			if(Source != source) {
				Source = source;
				LoadDirectory(null);
			}
		}

		private void LoadDirectory(string directory) {
			if(directory == null) {
				LoadItems(new string[0]);
			} else {
				directory = Application.dataPath + "/" + directory;
				if(Directory.Exists(directory)) {
					List<string> paths = new List<string>();
					Iterate(directory);
					LoadItems(paths.ToArray());
					void Iterate(string folder) {
						DirectoryInfo info = new DirectoryInfo(folder);
						foreach(FileInfo i in info.GetFiles()) {
							string path = i.FullName.Substring(i.FullName.IndexOf("Assets"));
							if((MotionAsset)AssetDatabase.LoadAssetAtPath(path, typeof(MotionAsset))) {
								paths.Add(path);
							}
						}
						Resources.UnloadUnusedAssets();
						foreach(DirectoryInfo i in info.GetDirectories()) {
							Iterate(i.FullName);
						}
					}
				} else {
					LoadItems(new string[0]);
				}
			}
		}

        public override void DerivedInspector(Item item) {
        
        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart()
        {
	        modifyFramerateResult = modifyFramerate;
	        modifyModelResult = modifyModel;
	        modifyTranslationResult = modifyTranslation;
	        modifyRotationResult = modifyRotation;
	        modifyScaleResult = modifyScale;
	        modifyMirrorAxisResult = modifyMirrorAxis;
	        modifyExportResult = modifyExport;
	        setAlignmentResult = setAlignment;
        }

        public override IEnumerator DerivedProcess(Item item) {
            MotionAsset asset = AssetDatabase.LoadAssetAtPath<MotionAsset>(item.ID);
            if(modifyFramerateResult)
				asset.Framerate = Framerate;
            if (modifyModelResult && !string.IsNullOrEmpty(Model))
				asset.Model = Model;
            if(modifyTranslationResult)
				asset.Translation = Translation;
            if(modifyRotationResult)
				asset.Rotation = Rotation;
            if(modifyScaleResult && Scale != 0)
				asset.Scale = Scale;
            if(modifyMirrorAxisResult)
				asset.MirrorAxis = MirrorAxis;
            if(modifyExportResult)
				asset.Export = Export;
            if (setAlignmentResult && standard != null)
            {
	            if (standard.Source.Bones.Length != asset.Source.Bones.Length)
	            {
		            Debug.LogError("Asset to modify has different bones num with Standard Motion Asset.");
		            yield break;
	            }
	            for(int i=0; i<asset.Source.Bones.Length; i++) {
		            asset.SetSymmetry(i, standard.Symmetry[i]);
		            asset.Source.Bones[i].Parent = standard.Source.Bones[i].Parent; 
		            asset.Source.Bones[i].Alignment = standard.Source.Bones[i].Alignment;
		            asset.Source.Bones[i].Correction = standard.Source.Bones[i].Correction;
		            asset.Source.Bones[i].Override = standard.Source.Bones[i].Override;
	            }
            }
            
            EditorUtility.SetDirty(asset);
            yield return new WaitForSeconds(0);
        }

        public override void BatchCallback() {
			AssetDatabase.SaveAssets();
			Resources.UnloadUnusedAssets();
        }

        public override void DerivedFinish() {

        }

    }
}
#endif
