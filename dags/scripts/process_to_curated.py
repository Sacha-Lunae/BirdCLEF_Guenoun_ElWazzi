import io
import os
import numpy as np
from pymongo import MongoClient
import gridfs
from minio import Minio
from minio.error import S3Error

# Global parameters
THRESHOLD = 2.2
BATCH_SIZE = 50  # Adjust based on available RAM

#####################################
# Spectrogram Processing Functions
#####################################

def spectral_gate(S, noise_mean, threshold=THRESHOLD):
    """
    Apply spectral gating: retains values greater than noise_mean * threshold.
    """
    mask = S > (noise_mean[:, np.newaxis] * threshold)
    S_denoised = S * mask
    return S_denoised

def apply_distortion(S, factor=1.1):
    """
    Apply a simple distortion by raising the spectrogram values to the power of 'factor'.
    """
    S_distorted = np.power(S, factor)
    return S_distorted

def process_spectrogram(S, threshold=THRESHOLD, apply_distortion_flag=True):
    """
    Estimate a noise profile (10th percentile) from the base spectrogram S,
    apply spectral gating and optionally a distortion.
    Returns a tuple (S_denoised, S_distorted).
    """
    noise_mean = np.percentile(S, 10, axis=1)
    S_denoised = spectral_gate(S, noise_mean, threshold)
    if apply_distortion_flag:
        S_distorted = apply_distortion(S_denoised)
    else:
        S_distorted = S_denoised
    return S_denoised, S_distorted

def compute_quality_metric(S):
    """
    Compute a simple audio quality metric based on a signal-to-noise ratio (SNR)-like calculation.
    Here we compute the average signal energy and a noise estimate using the 10th percentile across frequencies.
    Returns the SNR.
    """
    signal_energy = np.mean(S)
    noise_estimate = np.mean(np.percentile(S, 10, axis=1))
    snr = signal_energy / (noise_estimate + 1e-10)  # add epsilon to avoid division by zero
    return snr

def create_minio_client(endpoint, access_key, secret_key, secure=False):
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

##############################################
# Curation Pipeline with GridFS
##############################################

def curate_metadata_with_spectrograms_batch():
    # Connect to the source metadata database
    source_client = MongoClient("mongodb://mongodb:27017")
    source_db = source_client["birdclef"]   # Adjust as necessary
    metadata_coll = source_db["metadata"]

    # Connect to the "curated" database for enriched documents
    curated_client = MongoClient("mongodb://mongodb:27017")
    curated_db = curated_client["birdclef"]
    curated_coll = curated_db["curated_data"]
    
    # Create a GridFS object in the "spectrograms" collection of the curated database
    fs = gridfs.GridFS(curated_db, collection="spectrograms")

    # Connect to the staging bucket in Minio
    staging_endpoint = "minio:9000"   # S3 port for Minio
    staging_access_key = "minioadmin"
    staging_secret_key = "minioadmin"
    staging_bucket = "staging-bucket"
    minio_client = create_minio_client(staging_endpoint, staging_access_key, staging_secret_key, secure=False)

    # Iterate over the documents without loading the entire collection into memory
    cursor = metadata_coll.find({})
    batch_docs = []
    count_total = 0
    count_processed = 0

    for doc in cursor:
        count_total += 1
        # Remove unwanted fields and the _id field
        curated_doc = {k: v for k, v in doc.items() if k not in ["license", "rating", "author", "_id"]}
        
        filename = doc.get("filename")
        if not filename:
            print("Document without filename, skipping...")
            continue
        
        # Example: "asbfly/XC134896.ogg" → base_name = "XC134896"
        base_name = os.path.splitext(os.path.basename(filename))[0]
        npy_filename = f"{base_name}.npy"

        # Download the .npy file from the staging bucket
        try:
            response = minio_client.get_object(staging_bucket, npy_filename)
            npy_data = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            print(f"Minio error for {npy_filename}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error for {npy_filename}: {e}")
            continue

        try:
            npy_buffer = io.BytesIO(npy_data)
            spec_base = np.load(npy_buffer)
        except Exception as e:
            print(f"Error loading {npy_filename}: {e}")
            continue

        # Process the spectrogram: denoising and optional distortion
        try:
            spec_denoised, spec_distorted = process_spectrogram(spec_base, threshold=THRESHOLD, apply_distortion_flag=True)
        except Exception as e:
            print(f"Error processing spectrogram {npy_filename}: {e}")
            continue

        # Compute audio quality metric and determine "is_quality_audio"
        try:
            snr = compute_quality_metric(spec_base)
            # Using a heuristic threshold; adjust based on your testing/requirements
            curated_doc["is_quality_audio"] = bool(snr > 3)
        except Exception as e:
            print(f"Error computing quality metric for {npy_filename}: {e}")
            curated_doc["is_quality_audio"] = False

        # Store each spectrogram version in GridFS
        try:
            # Store the base version
            base_buffer = io.BytesIO()
            np.save(base_buffer, spec_base, allow_pickle=False)
            base_buffer.seek(0)
            base_file_id = fs.put(base_buffer.getvalue(),
                                  filename=f"{base_name}_base.npy",
                                  contentType="application/octet-stream")
            
            # Store the denoised version
            denoised_buffer = io.BytesIO()
            np.save(denoised_buffer, spec_denoised, allow_pickle=False)
            denoised_buffer.seek(0)
            denoised_file_id = fs.put(denoised_buffer.getvalue(),
                                      filename=f"{base_name}_denoised.npy",
                                      contentType="application/octet-stream")
            
            # Store the distorted version
            distorted_buffer = io.BytesIO()
            np.save(distorted_buffer, spec_distorted, allow_pickle=False)
            distorted_buffer.seek(0)
            distorted_file_id = fs.put(distorted_buffer.getvalue(),
                                       filename=f"{base_name}_distorted.npy",
                                       contentType="application/octet-stream")
        except Exception as e:
            print(f"GridFS storage error for {npy_filename}: {e}")
            continue

        # Add the GridFS references to the document
        curated_doc["spectrogram_base_id"] = base_file_id
        curated_doc["spectrogram_denoised_id"] = denoised_file_id
        curated_doc["spectrogram_distorted_id"] = distorted_file_id
        
        batch_docs.append(curated_doc)
        count_processed += 1
        print(f"Document {base_name} processed ({count_processed} documents processed).")
        
        # Batch insertion to limit memory usage
        if len(batch_docs) >= BATCH_SIZE:
            curated_coll.insert_many(batch_docs)
            print(f"{len(batch_docs)} documents inserted into 'birdclef.curated_data'.")
            batch_docs = []  # Reset for next batch

    # Insert the final batch, if any
    if batch_docs:
        curated_coll.insert_many(batch_docs)
        print(f"{len(batch_docs)} documents inserted into 'birdclef.curated_data' (final batch).")

    print(f"Processing completed. Documents processed: {count_processed} out of {count_total}.")

def main():
    curate_metadata_with_spectrograms_batch()

if __name__ == "__main__":
    main()
