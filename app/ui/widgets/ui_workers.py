import uuid
from functools import partial
from typing import TYPE_CHECKING, Dict
import traceback
import os

import cv2
import torch
import numpy
from PySide6 import QtCore as qtc
from PySide6.QtGui import QPixmap

from app.processors.models_data import detection_model_mapping, landmark_model_mapping
from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA, CAMERA_BACKENDS

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

class TargetMediaLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, QPixmap, str, str)  # Signal with media path and QPixmap and file_type, media_id
    webcam_thumbnail_ready = qtc.Signal(str, QPixmap, str, str, int, int)
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(self, main_window: 'MainWindow', folder_name=False, files_list=None, media_ids=None, webcam_mode=False, parent=None,):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.media_ids = media_ids or []
        self.webcam_mode = webcam_mode
        self._running = True  # Flag to control the running state
        
        # Ensure thumbnail directory exists
        misc_helpers.ensure_thumbnail_dir()

    def run(self):
        if self.folder_name:
            self.load_videos_and_images_from_folder(self.folder_name)
        if self.files_list:
            self.load_videos_and_images_from_files_list(self.files_list)
        if self.webcam_mode:
            self.load_webcams()
        self.finished.emit()

    def load_videos_and_images_from_folder(self, folder_name):
        # Initially hide the placeholder text
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        video_files = misc_helpers.get_video_files(folder_name, self.main_window.control['TargetMediaFolderRecursiveToggle'])
        image_files = misc_helpers.get_image_files(folder_name, self.main_window.control['TargetMediaFolderRecursiveToggle'])

        i=0
        media_files = video_files + image_files
        for media_file in media_files:
            if not self._running:  # Check if the thread is still running
                break
            media_file_path = os.path.join(folder_name, media_file)
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path, file_type)
            if self.media_ids:
                media_id = self.media_ids[i]
            else:
                media_id = str(uuid.uuid1().int)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type, media_id)
            i+=1
        # Show/Hide the placeholder text based on the number of items in ListWidget
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def load_videos_and_images_from_files_list(self, files_list):
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        media_files = files_list
        i=0
        for media_file_path in media_files:
            if not self._running:  # Check if the thread is still running
                break
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path, file_type=file_type)
            if self.media_ids:
                media_id = self.media_ids[i]
            else:
                media_id = str(uuid.uuid1().int)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type,media_id)
            i+=1
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def load_webcams(self,):
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        camera_backend = CAMERA_BACKENDS[self.main_window.control['WebcamBackendSelection']]
        for i in range(int(self.main_window.control['WebcamMaxNoSelection'])):
            try:
                pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path=f'Webcam {i}', file_type='webcam', webcam_index=i, webcam_backend=camera_backend)
                media_id = str(uuid.uuid1().int)

                if pixmap:
                    # Emit the signal to update GUI
                    self.webcam_thumbnail_ready.emit(f'Webcam {i}', pixmap, 'webcam',media_id, i, camera_backend)
            except Exception: # pylint: disable=broad-exception-caught
                traceback.print_exc()
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.wait()

class InputFacesLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, numpy.ndarray, object, QPixmap, str)
    finished = qtc.Signal()  # Signal to indicate completion
    def __init__(self, main_window: 'MainWindow', media_path=False, folder_name=False, files_list=None, face_ids=None,  parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.face_ids = face_ids or []
        self._running = True  # Flag to control the running state
        self.was_playing = True
        # НЕ вызываем pre_load_detection_recognition_models() здесь - вызовем в run() после запуска потока
        
    def pre_load_detection_recognition_models(self):
        import time
        t_start = time.time()
        print(f"[Startup] InputFacesLoaderWorker.pre_load_detection_recognition_models() started")
        control = self.main_window.control.copy()
        detect_model = detection_model_mapping[control['DetectorModelSelection']]
        landmark_detect_model = landmark_model_mapping[control['LandmarkDetectModelSelection']]
        models_processor = self.main_window.models_processor
        if self.main_window.video_processor.processing:
            was_playing = True
            self.main_window.buttonMediaPlay.click()
        else:
            was_playing = False
        t0 = time.time()
        if detect_model not in models_processor.models or models_processor.models[detect_model] is None:
            models_processor.models[detect_model] = models_processor.load_model(detect_model)
            print(f"[Startup] InputFacesLoaderWorker: loaded {detect_model} in {time.time() - t0:.2f}s")
        else:
            print(f"[Startup] InputFacesLoaderWorker: {detect_model} already loaded")
        t0 = time.time()
        if (landmark_detect_model not in models_processor.models or models_processor.models[landmark_detect_model] is None) and control['LandmarkDetectToggle']:
            models_processor.models[landmark_detect_model] = models_processor.load_model(landmark_detect_model)
            print(f"[Startup] InputFacesLoaderWorker: loaded {landmark_detect_model} in {time.time() - t0:.2f}s")
        elif control['LandmarkDetectToggle']:
            print(f"[Startup] InputFacesLoaderWorker: {landmark_detect_model} already loaded")
        t0 = time.time()
        loaded_count = 0
        for recognition_model in ['Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'CSCSArcFace', 'CSCSIDArcFace']:
            if recognition_model not in models_processor.models or models_processor.models[recognition_model] is None:
                t1 = time.time()
                models_processor.models[recognition_model] = models_processor.load_model(recognition_model)
                loaded_count += 1
                print(f"[Startup] InputFacesLoaderWorker: loaded {recognition_model} in {time.time() - t1:.2f}s")
        if loaded_count > 0:
            print(f"[Startup] InputFacesLoaderWorker: loaded {loaded_count} recognition models in {time.time() - t0:.2f}s")
        if was_playing:
            self.main_window.buttonMediaPlay.click()
        elapsed = time.time() - t_start
        print(f"[Startup] InputFacesLoaderWorker.pre_load_detection_recognition_models() completed in {elapsed:.2f}s")
        if hasattr(self.main_window, '_startup_start_time'):
            total_elapsed = time.time() - self.main_window._startup_start_time
            print(f"[Startup] pre_load_detection_recognition_models() total time since init: {total_elapsed:.2f}s")

    def run(self):
        import time
        t_start = time.time()
        print(f"[Startup] InputFacesLoaderWorker.run() started")
        # Загружаем модели детекции и распознавания перед использованием
        # Это гарантирует, что модели загружены в фоновом потоке после завершения ModelWarmupWorker
        self.pre_load_detection_recognition_models()
        if self.folder_name or self.files_list:
            self.main_window.placeholder_update_signal.emit(self.main_window.inputFacesList, True)
            t0 = time.time()
            self.load_faces(self.folder_name, self.files_list)
            print(f"[Startup] InputFacesLoaderWorker.load_faces() completed in {time.time() - t0:.2f}s")
            self.main_window.placeholder_update_signal.emit(self.main_window.inputFacesList, False)
        elapsed = time.time() - t_start
        print(f"[Startup] InputFacesLoaderWorker.run() completed in {elapsed:.2f}s")

    def load_faces(self, folder_name=False, files_list=None):
        import time
        t_start = time.time()
        print(f"[Startup] InputFacesLoaderWorker.load_faces() started")
        
        control = self.main_window.control.copy()
        models_processor = self.main_window.models_processor
        
        # Проверяем, что модель детекции загружена
        detect_model = detection_model_mapping[control['DetectorModelSelection']]
        if detect_model not in models_processor.models or models_processor.models[detect_model] is None:
            print(f"[Startup] ERROR: Detection model {detect_model} is not loaded! Loading now...")
            models_processor.models[detect_model] = models_processor.load_model(detect_model)
            print(f"[Startup] Detection model {detect_model} loaded successfully")
        else:
            print(f"[Startup] Detection model {detect_model} is ready")
        
        files_list = files_list or []
        image_files = []
        if folder_name:
            image_files = misc_helpers.get_image_files(self.folder_name, self.main_window.control['InputFacesFolderRecursiveToggle'])
            print(f"[Startup] Found {len(image_files)} image files in {folder_name}")
        elif files_list:
            image_files = files_list
            print(f"[Startup] Processing {len(image_files)} image files from list")

        i=0
        faces_found = 0
        faces_processed = 0
        image_files.sort()
        for image_file_path in image_files:
            if not self._running:  # Check if the thread is still running
                break
            if not misc_helpers.is_image_file(image_file_path):
                print(f"[Startup] Skipping non-image file: {image_file_path}")
                continue
            if folder_name:
                image_file_path = os.path.join(folder_name, image_file_path)
            
            t_file = time.time()
            frame = misc_helpers.read_image_file(image_file_path)
            if frame is None:
                print(f"[Startup] Failed to read image: {image_file_path}")
                continue
            print(f"[Startup] Loaded image {i+1}/{len(image_files)}: {os.path.basename(image_file_path)} in {time.time() - t_file:.2f}s")
            # Frame must be in RGB format
            frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

            img = torch.from_numpy(frame.astype('uint8')).to(self.main_window.models_processor.device)
            img = img.permute(2,0,1)
            
            t_detect = time.time()
            _, kpss_5, _ = self.main_window.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=1, score=control['DetectorScoreSlider']/100.0, input_size=(512, 512), use_landmark_detection=control['LandmarkDetectToggle'], landmark_detect_mode=control['LandmarkDetectModelSelection'], landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=control["DetectFromPointsToggle"], rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270])
            detect_time = time.time() - t_detect
            faces_found += 1
            
            # If atleast one face is found
            # found_face = []
            face_kps = False
            try:
                face_kps = kpss_5[0]
                print(f"[Startup] Face detected in {os.path.basename(image_file_path)} (detection took {detect_time:.2f}s)")
            except IndexError:
                print(f"[Startup] No face detected in {os.path.basename(image_file_path)} (detection took {detect_time:.2f}s)")
                continue
            if face_kps.any():
                print(f"[Startup] Face keypoints are valid, processing recognition for {os.path.basename(image_file_path)}")
                try:
                    t_rec = time.time()
                    face_emb, cropped_img = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                    print(f"[Startup] Main recognition completed in {time.time() - t_rec:.2f}s")
                    
                    cropped_img = cropped_img.cpu().numpy()
                    cropped_img = cropped_img[..., ::-1]  # Swap the channels from RGB to BGR
                    face_img = numpy.ascontiguousarray(cropped_img)
                    # crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                    pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, face_img)

                    embedding_store: Dict[str, numpy.ndarray] = {}
                    # Ottenere i valori di 'options'
                    options = SETTINGS_LAYOUT_DATA['Face Recognition']['RecognitionModelSelection']['options']
                    t_emb = time.time()
                    for option in options:
                        if option != control['RecognitionModelSelection']:
                            target_emb, _ = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], option)
                            embedding_store[option] = target_emb
                        else:
                            embedding_store[control['RecognitionModelSelection']] = face_emb
                    print(f"[Startup] All embeddings extracted in {time.time() - t_emb:.2f}s")
                    
                    if not self.face_ids:
                        face_id = str(uuid.uuid1().int)
                    else:
                        face_id = self.face_ids[i]
                    
                    self.thumbnail_ready.emit(image_file_path, face_img, embedding_store, pixmap, face_id)
                    faces_processed += 1
                    print(f"[Startup] Successfully processed face {faces_processed} from {os.path.basename(image_file_path)}")
                    i+=1
                except Exception as e:
                    print(f"[Startup] Error processing face from {os.path.basename(image_file_path)}: {e}")
                    traceback.print_exc()
                    continue
            else:
                print(f"[Startup] Face keypoints are empty/invalid for {os.path.basename(image_file_path)}")
        
        elapsed = time.time() - t_start
        print(f"[Startup] InputFacesLoaderWorker.load_faces() completed:")
        print(f"[Startup]   - Total images processed: {i}/{len(image_files)}")
        print(f"[Startup]   - Faces detected: {faces_found}")
        print(f"[Startup]   - Faces successfully processed: {faces_processed}")
        print(f"[Startup]   - Total time: {elapsed:.2f}s")
        
        torch.cuda.empty_cache()
        self.finished.emit()

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.wait()

class ModelWarmupWorker(qtc.QThread):
    finished = qtc.Signal()

    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window

    def run(self):
        import time
        t_start = time.time()
        print(f"[Startup] ModelWarmupWorker.run() started")
        try:
            control = self.main_window.control.copy()
            models_processor = self.main_window.models_processor

            t0 = time.time()
            detect_model = detection_model_mapping[control['DetectorModelSelection']]
            if not models_processor.models.get(detect_model):
                models_processor.models[detect_model] = models_processor.load_model(detect_model)
                print(f"[Startup] ModelWarmupWorker: loaded {detect_model} in {time.time() - t0:.2f}s")

            t0 = time.time()
            landmark_model = landmark_model_mapping[control['LandmarkDetectModelSelection']]
            if control.get('LandmarkDetectToggle') and not models_processor.models.get(landmark_model):
                models_processor.models[landmark_model] = models_processor.load_model(landmark_model)
                print(f"[Startup] ModelWarmupWorker: loaded {landmark_model} in {time.time() - t0:.2f}s")

            recognition_models = []

            def _collect(model_name):
                if model_name and model_name not in recognition_models:
                    recognition_models.append(model_name)

            _collect(control.get('RecognitionModelSelection'))

            swapper_model = control.get('SwapModelSelection')
            if swapper_model:
                try:
                    _collect(models_processor.get_arcface_model(swapper_model))
                    if swapper_model == 'CSCS':
                        _collect('CSCSIDArcFace')
                except Exception:
                    pass

            if not recognition_models:
                recognition_models = ['Inswapper128ArcFace']

            t0 = time.time()
            for recognition_model in recognition_models:
                if not models_processor.models.get(recognition_model):
                    t1 = time.time()
                    models_processor.models[recognition_model] = models_processor.load_model(recognition_model)
                    print(f"[Startup] ModelWarmupWorker: loaded {recognition_model} in {time.time() - t1:.2f}s")
            if recognition_models:
                print(f"[Startup] ModelWarmupWorker: loaded {len(recognition_models)} recognition models in {time.time() - t0:.2f}s")

            restorer_model_map = {
                'GFPGAN-v1.4': 'GFPGANv1.4',
                'CodeFormer': 'CodeFormer',
                'GPEN-256': 'GPENBFR256',
                'GPEN-512': 'GPENBFR512',
                'GPEN-1024': 'GPENBFR1024',
                'GPEN-2048': 'GPENBFR2048',
                'RestoreFormer++': 'RestoreFormerPlusPlus',
                'VQFR-v2': 'VQFRv2',
            }

            if control.get('FaceRestorerEnableToggle'):
                restorer_type = control.get('FaceRestorerTypeSelection')
                restorer_model = restorer_model_map.get(restorer_type)
                if restorer_model and not models_processor.models.get(restorer_model):
                    models_processor.models[restorer_model] = models_processor.load_model(restorer_model)

            if control.get('FaceRestorerEnable2Toggle'):
                restorer_type = control.get('FaceRestorerType2Selection')
                restorer_model = restorer_model_map.get(restorer_type)
                if restorer_model and not models_processor.models.get(restorer_model):
                    t1 = time.time()
                    models_processor.models[restorer_model] = models_processor.load_model(restorer_model)
                    print(f"[Startup] ModelWarmupWorker: loaded {restorer_model} in {time.time() - t1:.2f}s")
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
        finally:
            elapsed = time.time() - t_start
            print(f"[Startup] ModelWarmupWorker.run() completed in {elapsed:.2f}s")
            self.finished.emit()


class FilterWorker(qtc.QThread):
    filtered_results = qtc.Signal(list)

    def __init__(self, main_window: 'MainWindow', search_text='', filter_list='target_videos'):
        super().__init__()
        self.main_window = main_window
        self.search_text = search_text
        self.filter_list = filter_list
        self.filter_list_widget = self.get_list_widget()
        self.filtered_results.connect(partial(filter_actions.update_filtered_list, main_window, self.filter_list_widget))

    def get_list_widget(self,):
        list_widget = False
        if self.filter_list == 'target_videos':
            list_widget = self.main_window.targetVideosList
        elif self.filter_list == 'input_faces':
            list_widget = self.main_window.inputFacesList
        elif self.filter_list == 'merged_embeddings':
            list_widget = self.main_window.inputEmbeddingsList
        return list_widget

    def run(self,):
        if self.filter_list == 'target_videos':
            self.filter_target_videos(self.main_window, self.search_text)
        elif self.filter_list == 'input_faces':
            self.filter_input_faces(self.main_window, self.search_text)
        elif self.filter_list == 'merged_embeddings':
            self.filter_merged_embeddings(self.main_window, self.search_text)


    def filter_target_videos(self, main_window: 'MainWindow', search_text: str = ''):
        search_text = main_window.targetVideosSearchBox.text().lower()
        include_file_types = []
        if main_window.filterImagesCheckBox.isChecked():
            include_file_types.append('image')
        if main_window.filterVideosCheckBox.isChecked():
            include_file_types.append('video')
        if main_window.filterWebcamsCheckBox.isChecked():
            include_file_types.append('webcam')

        visible_indices = []
        for i in range(main_window.targetVideosList.count()):
            item = main_window.targetVideosList.item(i)
            item_widget = main_window.targetVideosList.itemWidget(item)
            if ((not search_text or search_text in item_widget.media_path.lower()) and 
                (item_widget.file_type in include_file_types)):
                visible_indices.append(i)

        self.filtered_results.emit(visible_indices)

    def filter_input_faces(self, main_window: 'MainWindow', search_text: str):
        search_text = search_text.lower()
        visible_indices = []

        for i in range(main_window.inputFacesList.count()):
            item = main_window.inputFacesList.item(i)
            item_widget = main_window.inputFacesList.itemWidget(item)
            if not search_text or search_text in item_widget.media_path.lower():
                visible_indices.append(i)

        self.filtered_results.emit(visible_indices)

    def filter_merged_embeddings(self, main_window: 'MainWindow', search_text: str):
        search_text = search_text.lower()
        visible_indices = []

        for i in range(main_window.inputEmbeddingsList.count()):
            item = main_window.inputEmbeddingsList.item(i)
            item_widget = main_window.inputEmbeddingsList.itemWidget(item)
            if not search_text or search_text in item_widget.embedding_name.lower():
                visible_indices.append(i)

        self.filtered_results.emit(visible_indices)

    def stop_thread(self):
        self.quit()
        self.wait()
