import yaml
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
from sklearn.metrics import silhouette_score

def load_configuration():
    """Loads the YAML configuration file."""
    print("--- Initializing System Configuration ---")
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print("✅ Configuration loaded successfully.")
    return config

def build_feature_matrix(target_folder_path):
    """Extracts features from raw images to build the mathematical matrix."""
    folder = Path(target_folder_path)
    print(f"--- Starting Feature Extraction: {folder.name} ---")
    
    # ALARM 1: Does VS Code actually see the folder?
    if not folder.exists():
        print(f"❌ CRITICAL ERROR: VS Code cannot find or access the folder at: {folder}")
        print("-> Check macOS System Settings > Privacy & Security > Files and Folders, and ensure VS Code has Desktop access.")
        return np.array([]), []
        
    features_dataset = []
    file_names = []
    
    for image_file in folder.glob("*.png"):
        try:
            img = Image.open(image_file).resize((512, 512))
            img_array = np.array(img)
            
            avg_density = np.mean(img_array)
            contrast = np.std(img_array)
            peak_value = np.max(img_array)
            
            features_dataset.append([avg_density, contrast, peak_value])
            file_names.append(image_file.name)
        except Exception as e:
            # ALARM 2: Stop skipping errors silently! Print the exact reason.
            print(f"⚠️ Failed to process {image_file.name}. Reason: {e}")
            continue
            
    X_features = np.array(features_dataset)
    print(f"✅ Extraction Complete. Feature Matrix Shape: {X_features.shape}")
    return X_features, file_names

def main():
    """The main engine that runs the pipeline top-to-bottom."""
    print("\n========== PROJECT MAMMO: QC PIPELINE ==========\n")
    
    # 1. Load Settings
    config = load_configuration()
    DATA_FOLDER = config['paths']['data_folder']
    N_COMPONENTS = config['model']['n_components']
    SAFETY_THRESHOLD = config['business_rules']['safety_threshold']
    
    # 2. Extract Features (This creates X and names)
    X, names = build_feature_matrix(DATA_FOLDER)
    
    if len(X) == 0:
        print("❌ Error: No images processed. Check your folder path.")
        return

    # 3. Train the Probabilistic AI
    print("--- Scaling Features & Training GMM AI ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42)
    gmm_clusters = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    print("✅ AI Training Complete.")
    
    print("--- Scaling Features & Training GMM AI ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42)
    gmm_clusters = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    # NEW CODE: Calculate the Silhouette Score
    sil_score = silhouette_score(X_scaled, gmm_clusters)
    print(f"✅ AI Training Complete. Silhouette Score: {sil_score:.3f}")

    # 4. The Triage Engine & Visualization
    print(f"\n--- Automated QC Engine & Triage ---")
    
    # Create our sorting buckets
    triage_bins = {
        "Excellent (90-100%)": [],
        "Good (75-89%)": [],
        "Borderline (50-74%)": [],
        "Critical (<50%)": []
    }

    flagged_count = 0
    plt.figure(figsize=(12, 7))
    plt.scatter(X[:, 0], X[:, 1], c=gmm_clusters, cmap='viridis', alpha=0.5, s=60)

    # Loop through and sort every single image
    for i in range(len(names)):
        max_confidence = np.max(probabilities[i]) * 100
        file_name = names[i]
        
        # Sort into buckets
        if max_confidence >= 90:
            triage_bins["Excellent (90-100%)"].append((file_name, max_confidence))
        elif max_confidence >= SAFETY_THRESHOLD:
            triage_bins["Good (75-89%)"].append((file_name, max_confidence))
        elif max_confidence >= 50:
            triage_bins["Borderline (50-74%)"].append((file_name, max_confidence))
        else:
            triage_bins["Critical (<50%)"].append((file_name, max_confidence))

        # Flag and paint the anomalies red on the chart
        if max_confidence < SAFETY_THRESHOLD:
            plt.scatter(X[i, 0], X[i, 1], color='red', edgecolor='darkred', s=90, zorder=5)
            flagged_count += 1

   # 5. Render the Executive Scatter Plot (Cleaned up!)
    plt.title('Project Mammo QC: Probabilistic Clustering\n(Red = AI requires human review)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Feature 1: Average Tissue Density', fontsize=12)
    plt.ylabel('Feature 2: Image Contrast', fontsize=12)

    red_patch = mpatches.Patch(color='red', label=f'Flagged Anomalies (<{SAFETY_THRESHOLD}%)')
    standard_patch = mpatches.Patch(color='gray', label='Verified Scans')
    plt.legend(handles=[red_patch, standard_patch], loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show() # The script pauses here until you close the scatter plot

    # ==========================================
    # 6. NEW: THE RADIOLOGIST REVIEW GRID
    # ==========================================
    print("\n--- Generating Visual Grid for Critical Scans ---")
    
    # Grab all flagged images and sort them to find the absolute worst ones
    flagged_list = []
    for i in range(len(names)):
        max_conf = np.max(probabilities[i]) * 100
        if max_conf < SAFETY_THRESHOLD:
            flagged_list.append((names[i], max_conf))

    # Sort so the lowest confidence scores are at the very beginning
    flagged_list.sort(key=lambda x: x[1])
    worst_scans = flagged_list[:9] # Grab the top 9 worst offenders

    # Only draw the grid if we actually caught anomalies
    if len(worst_scans) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle('MedSendX Critical Anomalies (Lowest AI Confidence)', fontsize=16, fontweight='bold')

        # Flatten the 3x3 axis array so we can loop through it easily
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < len(worst_scans):
                file_name, conf = worst_scans[idx]
                img_path = Path(DATA_FOLDER) / file_name
                
                try:
                    # Open the original image and display it
                    img = Image.open(img_path)
                    ax.imshow(img, cmap='gray') # Medical images look best in pure grayscale
                    ax.set_title(f"{file_name}\nAI Confidence: {conf:.1f}%", color='darkred', fontweight='bold', fontsize=10)
                except Exception as e:
                    ax.set_title("Image Load Error")
            
            # Turn off the graph borders and numbers for a clean photo-grid look
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
