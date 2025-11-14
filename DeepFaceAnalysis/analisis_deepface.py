"""
Script para analizar emociones en video usando DeepFace.
Extrae frames del video a intervalos regulares y analiza las emociones detectadas en cada frame.
"""

import cv2
from deepface import DeepFace
import os
import json
from datetime import datetime

def calculate_individual_emotion_averages(results):
    """
    Calcula el promedio de cada emoción a través de todos los frames analizados.
    
    Args:
        results: Lista de resultados del análisis de frames
        
    Returns:
        Diccionario con el promedio de cada emoción
    """
    if not results:
        return {}
    
    # Filtrar solo resultados exitosos
    successful_results = [r for r in results if r['dominant_emotion'] != 'error' and 'all_emotions' in r]
    
    if not successful_results:
        return {}
    
    # Obtener todas las emociones únicas
    all_emotions = set()
    for result in successful_results:
        if result['all_emotions']:
            all_emotions.update(result['all_emotions'].keys())
    
    if not all_emotions:
        return {}
    
    # Calcular sumas y contadores
    emotion_sums = {emotion: 0.0 for emotion in all_emotions}
    emotion_counts = {emotion: 0 for emotion in all_emotions}
    
    for result in successful_results:
        if result['all_emotions']:
            for emotion, score in result['all_emotions'].items():
                emotion_sums[emotion] += score
                emotion_counts[emotion] += 1
    
    # Calcular promedios
    emotion_averages = {}
    for emotion in all_emotions:
        if emotion_counts[emotion] > 0:
            emotion_averages[emotion] = emotion_sums[emotion] / emotion_counts[emotion]
    
    return emotion_averages

def analyze_video_emotions(video_path, frame_interval=1.0, save_frames=False):
    """
    Analiza emociones en un video frame por frame usando DeepFace.
    
    Args:
        video_path: Ruta al archivo de video (modificar según ubicación del video)
        frame_interval: Intervalo entre frames a analizar en segundos
        save_frames: Si True, guarda los frames extraídos en disco
        
    Returns:
        Diccionario con información del análisis y resultados detallados
    """
    
    if not os.path.exists(video_path):
        print(f"Error: No se encontró el video {video_path}")
        return None
    
    print(f"Analizando video: {video_path}")
    print(f"Intervalo entre frames: {frame_interval} segundos")
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        return None
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Información del video:")
    print(f"   Duración: {duration:.2f} segundos")
    print(f"   FPS: {fps:.2f}")
    print(f"   Resolución: {width}x{height}")
    print(f"   Total frames: {total_frames}")
    
    # Calcular frames a analizar según el intervalo
    frame_step = int(fps * frame_interval)
    estimated_frames = int(duration / frame_interval)
    print(f"   Frames a analizar: ~{estimated_frames}")
    
    # Crear directorio para frames si es necesario
    frames_dir = None
    if save_frames:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        frames_dir = f"frames_video_{timestamp}"
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Frames se guardarán en: {frames_dir}")
    
    print(f"\nIniciando análisis...")
    
    results = []
    frame_count = 0
    analyzed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analizar frame según el intervalo configurado
        if frame_count % frame_step == 0:
            timestamp = frame_count / fps
            analyzed_count += 1
            
            print(f"Frame {analyzed_count} - Tiempo: {timestamp:.1f}s", end=" ... ")
            
            try:
                # Guardar frame temporalmente para análisis
                temp_frame_name = f"temp_frame_{timestamp:.1f}s.jpg"
                cv2.imwrite(temp_frame_name, frame)
                
                # Analizar con DeepFace
                result = DeepFace.analyze(
                    img_path=temp_frame_name,
                    actions=['emotion'],
                    enforce_detection=False
                )
                
                # Procesar resultado de DeepFace
                if isinstance(result, list):
                    result = result[0]
                
                # Convertir scores a float para serialización JSON
                emotion_scores = {emotion: float(score) for emotion, score in result['emotion'].items()}
                
                frame_data = {
                    'frame_number': analyzed_count,
                    'timestamp': float(timestamp),
                    'dominant_emotion': result['dominant_emotion'],
                    'confidence': float(result['emotion'][result['dominant_emotion']]),
                    'all_emotions': emotion_scores
                }
                
                results.append(frame_data)
                print(f"{result['dominant_emotion']} ({result['emotion'][result['dominant_emotion']]:.1f}%)")
                
                # Guardar frame con metadata si se configuró
                if save_frames and frames_dir:
                    final_frame_name = f"frame_{analyzed_count:03d}_t{timestamp:.1f}s_{result['dominant_emotion']}.jpg"
                    final_frame_path = os.path.join(frames_dir, final_frame_name)
                    cv2.imwrite(final_frame_path, frame)
                
                # Eliminar frame temporal
                os.remove(temp_frame_name)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                frame_data = {
                    'frame_number': analyzed_count,
                    'timestamp': float(timestamp),
                    'dominant_emotion': 'error',
                    'confidence': 0.0,
                    'all_emotions': {},
                    'error': str(e)
                }
                results.append(frame_data)
                
                # Limpiar frame temporal si existe
                if os.path.exists(temp_frame_name):
                    os.remove(temp_frame_name)
        
        frame_count += 1
    
    cap.release()
    
    # Generar estadísticas y reportes
    if results:
        successful_results = [r for r in results if r['dominant_emotion'] != 'error']
        
        print(f"\nANÁLISIS COMPLETADO")
        print(f"   Frames procesados: {len(results)}")
        print(f"   Análisis exitosos: {len(successful_results)}")
        print(f"   Errores: {len(results) - len(successful_results)}")
        
        if successful_results:
            # Calcular distribución de emociones dominantes
            emotions = [r['dominant_emotion'] for r in successful_results]
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print(f"\nDISTRIBUCIÓN DE EMOCIONES:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(successful_results)) * 100
                print(f"   {emotion}: {count} frames ({percentage:.1f}%)")
            
            most_common_emotion = max(emotion_counts, key=emotion_counts.get)
            print(f"\nEmoción más frecuente: {most_common_emotion}")
            
            # Calcular promedios de cada emoción
            emotion_averages = calculate_individual_emotion_averages(results)
            
            if emotion_averages:
                print(f"\nPROMEDIOS DE EMOCIONES:")
                sorted_averages = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
                for emotion, avg_score in sorted_averages:
                    print(f"   {emotion}: {avg_score:.1f}%")
                
                highest_average_emotion = sorted_averages[0][0]
                print(f"\nMayor promedio: {highest_average_emotion} ({sorted_averages[0][1]:.1f}%)")
            else:
                highest_average_emotion = "N/A"
                emotion_averages = {}
            
            # Guardar resultados en archivos
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Archivo JSON con datos completos
            json_file = f"analisis_video_{timestamp}.json"
            video_analysis = {
                'video_info': {
                    'path': video_path,
                    'duration': float(duration),
                    'fps': float(fps),
                    'resolution': f"{width}x{height}",
                    'total_frames': total_frames
                },
                'analysis_settings': {
                    'frame_interval': frame_interval,
                    'frames_analyzed': len(successful_results),
                    'save_frames': save_frames,
                    'frames_directory': frames_dir
                },
                'results_summary': {
                    'most_frequent_emotion': most_common_emotion,
                    'highest_average_emotion': highest_average_emotion,
                    'emotion_distribution': emotion_counts,
                    'emotion_averages': emotion_averages,
                    'analysis_date': datetime.now().isoformat()
                },
                'detailed_results': results
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(video_analysis, f, indent=2, ensure_ascii=False)
            
            # Archivo de texto con reporte legible
            txt_file = f"reporte_video_{timestamp}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("ANÁLISIS DE EMOCIONES EN VIDEO\n")
                f.write("="*40 + "\n\n")
                f.write(f"Video: {video_path}\n")
                f.write(f"Duración: {duration:.2f} segundos\n")
                f.write(f"Frames analizados: {len(successful_results)}\n")
                f.write(f"Intervalo: cada {frame_interval} segundo(s)\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"MÉTODO 1 - EMOCIÓN MÁS FRECUENTE: {most_common_emotion.upper()}\n")
                f.write(f"MÉTODO 2 - MAYOR PROMEDIO: {highest_average_emotion.upper()}\n\n")
                
                f.write("DISTRIBUCIÓN DE EMOCIONES (Frecuencia):\n")
                f.write("-" * 40 + "\n")
                for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(successful_results)) * 100
                    f.write(f"{emotion.capitalize():12}: {count:3d} frames ({percentage:5.1f}%)\n")
                
                if emotion_averages:
                    f.write("\nPROMEDIOS DE EMOCIONES:\n")
                    f.write("-" * 40 + "\n")
                    sorted_averages = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
                    for emotion, avg_score in sorted_averages:
                        f.write(f"{emotion.capitalize():12}: {avg_score:5.1f}%\n")
                
                f.write(f"\nTIMELINE DE EMOCIONES:\n")
                f.write("-" * 30 + "\n")
                for result in successful_results:
                    f.write(f"{result['timestamp']:6.1f}s: {result['dominant_emotion']:10} ({result['confidence']:5.1f}%)\n")
            
            print(f"\nARCHIVOS GUARDADOS:")
            print(f"   Datos completos: {json_file}")
            print(f"   Reporte legible: {txt_file}")
            if frames_dir:
                print(f"   Frames guardados: {frames_dir}/")
            
            return video_analysis
        
        else:
            print("No se detectaron emociones en ningún frame")
            return None
    
    else:
        print("No se procesó ningún frame")
        return None

def main():
    """
    Función principal para ejecutar el análisis de video.
    """
    print("ANÁLISIS DE EMOCIONES EN VIDEO CON DEEPFACE")
    print("="*50)
    
    # Buscar video (modificar según ubicación del proyecto)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_filename = "5.mp4"
    
    # Buscar en múltiples ubicaciones posibles
    possible_paths = [
        video_filename,
        os.path.join(script_dir, video_filename),
        os.path.join("DeepFace", video_filename),
        os.path.join("..", "DeepFace", video_filename)
    ]
    
    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path:
        print(f"Video encontrado: {video_path}")
    else:
        print(f"No se encontró el video: {video_filename}")
        print("\nBuscando otros videos...")
        video_path = input("Ruta a otro video: ").strip().strip('"')
        
        if not video_path or not os.path.exists(video_path):
            print("Video no encontrado")
            return
    
    # Configuración del análisis
    print(f"\nCONFIGURACIONES:")
    
    interval_input = input("Intervalo entre frames (segundos, default 1.0): ").strip()
    interval = float(interval_input) if interval_input else 1.0
    
    save_input = input("¿Guardar frames extraídos? (s/n, default n): ").strip().lower()
    save_frames = save_input == 's'
    
    print(f"\nIniciando análisis...")
    print(f"   Video: {video_path}")
    print(f"   Intervalo: {interval} segundos")
    print(f"   Guardar frames: {'Sí' if save_frames else 'No'}")
    
    # Ejecutar análisis de video
    result = analyze_video_emotions(video_path, interval, save_frames)
    
    if result:
        print(f"\nAnálisis completado exitosamente")
        print(f"Revisa los archivos generados para ver los resultados detallados")
        
        # Resumen rápido
        summary = result['results_summary']
        print(f"\nRESUMEN RÁPIDO:")
        print(f"   Emoción más frecuente: {summary['most_frequent_emotion']}")
        print(f"   Mayor promedio individual: {summary['highest_average_emotion']}")
        print(f"   Frames analizados: {result['analysis_settings']['frames_analyzed']}")
        print(f"   Duración video: {result['video_info']['duration']:.1f} segundos")
    else:
        print(f"\nEl análisis no se pudo completar")

if __name__ == "__main__":
    main()
