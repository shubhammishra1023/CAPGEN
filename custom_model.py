import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Concatenate

def build_paper_model(vocab_size, max_length):
    """
    Implements the architecture described in Figure 1 & 3.1.8 of the paper.
    1. Visual Encoder: ResNet50 
    2. Sentiment Branch: Visual Mood [cite: 92]
    3. Decoder: LSTM [cite: 43]
    """
    
    # --- 1. Visual Encoder (The Eye) ---
    # Input image size 224x224 [cite: 62]
    inputs1 = Input(shape=(2048,)) # Output from ResNet50 pooling layer
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # --- 2. Sentiment Branch (The Feeler) ---
    # According to Section 3.1.4, Sentiment Vector is fused [cite: 95]
    # We assume a sentiment vector input (e.g., probability of Pos/Neg/Neut)
    inputs2 = Input(shape=(3,)) # Sentiment vector (Positive, Negative, Neutral)
    se1 = Dense(32, activation='relu')(inputs2) # Projecting sentiment to embedding space

    # --- 3. Feature Fusion (The Combiner) ---
    # merging Visual Vectors + Sentiment Vectors [cite: 70]
    fusion = Concatenate()([fe2, se1])
    fusion_processed = Dense(256, activation='relu')(fusion)

    # --- 4. Caption Decoder (The Writer) ---
    # Sequence input
    inputs3 = Input(shape=(max_length,))
    se2 = Embedding(vocab_size, 256, mask_zero=True)(inputs3)
    se3 = Dropout(0.5)(se2)
    
    # LSTM initializes with fused features [cite: 100]
    decoder1 = LSTM(256)(se3, initial_state=[fusion_processed, fusion_processed])
    
    # Output Layer
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Create Model
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam') # [cite: 44]
    
    return model

def extract_features_resnet(image_path):
    """
    Extracts features using ResNet50 as per Section 3.1.1 [cite: 52]
    """
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features