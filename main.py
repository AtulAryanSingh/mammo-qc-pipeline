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

# ==========================================
    # 5. Render the Executive Dashboard Plot
    # ==========================================
    plt.figure(figsize=(13, 8)) # Made it slightly wider to fit the report box
    
    # We will plot the dots layer by layer based on their Triage category
    # Extract the X and Y coordinates for each category from our triage_bins
    
    exc_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Excellent (90-100%)", [])]
    exc_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Excellent (90-100%)", [])]
    
    good_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Good (75-89%)", [])]
    good_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Good (75-89%)", [])]
    
    bord_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Borderline (50-74%)", [])]
    bord_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Borderline (50-74%)", [])]
    
    crit_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Critical (<50%)", [])]
    crit_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Critical (<50%)", [])]

    # Plot each category with its own distinct color and label
    if exc_x: plt.scatter(exc_x, exc_y, color='#2ca02c', label='Excellent (90-100%)', alpha=0.6, s=60)
    if good_x: plt.scatter(good_x, good_y, color='#1f77b4', label='Good (75-89%)', alpha=0.6, s=60)
    if bord_x: plt.scatter(bord_x, bord_y, color='#ff7f0e', label='Borderline (50-74%)', alpha=0.9, s=70)
    if crit_x: plt.scatter(crit_x, crit_y, color='#d62728', edgecolor='darkred', label='Critical (<50%)', s=100, zorder=5)

    plt.title('Project Mammo QC: Automated Triage Dashboard', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Feature 1: Average Tissue Density', fontsize=12)
    plt.ylabel('Feature 2: Image Contrast', fontsize=12)

    # Add the Legend
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Inject the Live Text Report directly onto the chart
    report_text = (
        "📊 QA Triage Report\n"
        "------------------------\n"
        f"Excellent Scans: {len(exc_x)}\n"
        f"Good Scans:       {len(good_x)}\n"
        f"Borderline:       {len(bord_x)}\n"
        f"Critical:          {len(crit_x)}\n"
        "------------------------\n"
        f"Total Processed:  {len(names)}"
    )
    
    # Place the text box in the top right corner
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, edgecolor='gray')
    plt.text(0.97, 0.96, report_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show() # Pauses here until you close the window
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

# ==========================================
    # 5. Render the Executive Dashboard Plot
    # ==========================================
    plt.figure(figsize=(13, 8)) # Made it slightly wider to fit the report box
    
    # We will plot the dots layer by layer based on their Triage category
    # Extract the X and Y coordinates for each category from our triage_bins
    
    exc_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Excellent (90-100%)", [])]
    exc_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Excellent (90-100%)", [])]
    
    good_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Good (75-89%)", [])]
    good_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Good (75-89%)", [])]
    
    bord_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Borderline (50-74%)", [])]
    bord_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Borderline (50-74%)", [])]
    
    crit_x = [X[names.index(name), 0] for name, _ in triage_bins.get("Critical (<50%)", [])]
    crit_y = [X[names.index(name), 1] for name, _ in triage_bins.get("Critical (<50%)", [])]

    # Plot each category with its own distinct color and label
    if exc_x: plt.scatter(exc_x, exc_y, color='#2ca02c', label='Excellent (90-100%)', alpha=0.6, s=60)
    if good_x: plt.scatter(good_x, good_y, color='#1f77b4', label='Good (75-89%)', alpha=0.6, s=60)
    if bord_x: plt.scatter(bord_x, bord_y, color='#ff7f0e', label='Borderline (50-74%)', alpha=0.9, s=70)
    if crit_x: plt.scatter(crit_x, crit_y, color='#d62728', edgecolor='darkred', label='Critical (<50%)', s=100, zorder=5)

    plt.title('Project Mammo QC: Automated Triage Dashboard', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Feature 1: Average Tissue Density', fontsize=12)
    plt.ylabel('Feature 2: Image Contrast', fontsize=12)

    # Add the Legend
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Inject the Live Text Report directly onto the chart
    report_text = (
        "📊 QA Triage Report\n"
        "------------------------\n"
        f"Excellent Scans: {len(exc_x)}\n"
        f"Good Scans:       {len(good_x)}\n"
        f"Borderline:       {len(bord_x)}\n"
        f"Critical:          {len(crit_x)}\n"
        "------------------------\n"
        f"Total Processed:  {len(names)}"
    )
    
    # Place the text box in the top right corner
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, edgecolor='gray')
    plt.text(0.97, 0.96, report_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show() # Pauses here until you close the window
if __name__ == "__main__":
    main()
