"""
Dog Matchmaker: AI-Powered Breed Recommendation Chatbot
Streamlit App for DataCamp Competition

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="🐕 Dog Matchmaker AI",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Get image path function
def get_image_path(breed_name):
    """Get the image path for a breed from the cloned repository."""
    clean_name = breed_name.lower().strip()
    images_base = 'Dog-Breeds-Dataset/Images'
    
    if not os.path.exists(images_base):
        return None
    
    try:
        breed_dirs = [d for d in os.listdir(images_base) 
                     if os.path.isdir(os.path.join(images_base, d))]
        
        # Try to find matching directory
        for breed_dir in breed_dirs:
            breed_dir_lower = breed_dir.lower()
            
            # Check various matching strategies
            # 1. Direct substring match
            if clean_name.replace(' ', '') in breed_dir_lower.replace(' ', ''):
                dir_path = os.path.join(images_base, breed_dir)
                images = [f for f in os.listdir(dir_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                if images:
                    return os.path.join(dir_path, images[0])
            
            # 2. Check if any significant word matches
            words = [w for w in clean_name.split() if len(w) > 3]
            if any(word in breed_dir_lower for word in words):
                dir_path = os.path.join(images_base, breed_dir)
                images = [f for f in os.listdir(dir_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                if images:
                    return os.path.join(dir_path, images[0])
                    
    except Exception as e:
        print(f"Error loading image for {breed_name}: {e}")
    
    return None

# Dog Matchmaker Class
class DogMatchmaker:
    def __init__(self, breed_traits_df):
        self.breed_traits_original = breed_traits_df.copy()
        self.breed_traits = breed_traits_df.copy()
        
        # Get only numeric trait columns
        self.trait_columns = [col for col in breed_traits_df.columns 
                             if col not in ['Breed', 'Coat Type', 'Coat Length'] 
                             and breed_traits_df[col].dtype in ['int64', 'float64']]
        
        # Normalize traits
        scaler = MinMaxScaler()
        self.breed_traits[self.trait_columns] = scaler.fit_transform(
            self.breed_traits[self.trait_columns]
        )
    
    def create_user_profile(self, preferences):
        profile = {}
        weights = {}
        
        # Activity Level
        if 'activity_level' in preferences:
            activity = preferences['activity_level'].lower()
            if 'very active' in activity or 'high' in activity:
                profile['Energy Level'] = 5
                profile['Playfulness Level'] = 5
                weights['Energy Level'] = 2.0
                weights['Playfulness Level'] = 1.5
            elif 'moderate' in activity:
                profile['Energy Level'] = 3
                profile['Playfulness Level'] = 3
                weights['Energy Level'] = 1.5
                weights['Playfulness Level'] = 1.0
            elif 'low' in activity or 'calm' in activity or 'couch' in activity:
                profile['Energy Level'] = 2
                profile['Playfulness Level'] = 2
                weights['Energy Level'] = 2.0
                weights['Playfulness Level'] = 1.5
        
        # Living Space
        if 'living_space' in preferences:
            space = preferences['living_space'].lower()
            if 'apartment' in space or 'small' in space:
                profile['Adaptability Level'] = 5
                profile['Energy Level'] = 2
                weights['Adaptability Level'] = 2.0
            elif 'house' in space and 'yard' in space:
                profile['Energy Level'] = 4
                weights['Energy Level'] = 1.0
        
        # Children
        if 'children' in preferences:
            children = preferences['children'].lower()
            if 'yes' in children or 'young' in children:
                profile['Good With Young Children'] = 5
                profile['Affectionate With Family'] = 5
                weights['Good With Young Children'] = 3.0
                weights['Affectionate With Family'] = 1.5
            elif 'no' in children:
                weights['Good With Young Children'] = 0.1
        
        # Other Dogs
        if 'other_dogs' in preferences:
            other_dogs = preferences['other_dogs'].lower()
            if 'yes' in other_dogs:
                profile['Good With Other Dogs'] = 5
                weights['Good With Other Dogs'] = 3.0
            elif 'no' in other_dogs:
                weights['Good With Other Dogs'] = 0.1
        
        # Allergies/Shedding
        if 'allergies' in preferences:
            allergies = preferences['allergies'].lower()
            if 'yes' in allergies:
                profile['Shedding Level'] = 1
                weights['Shedding Level'] = 3.0
            else:
                weights['Shedding Level'] = 0.5
        
        # Grooming Time
        if 'grooming_time' in preferences:
            grooming = preferences['grooming_time'].lower()
            if 'low' in grooming or 'minimal' in grooming:
                profile['Coat Grooming Frequency'] = 1
                weights['Coat Grooming Frequency'] = 2.0
            elif 'moderate' in grooming:
                profile['Coat Grooming Frequency'] = 3
                weights['Coat Grooming Frequency'] = 1.0
            elif 'high' in grooming or 'lot' in grooming:
                weights['Coat Grooming Frequency'] = 0.5
        
        # Training Experience
        if 'training_experience' in preferences:
            experience = preferences['training_experience'].lower()
            if 'first' in experience or 'beginner' in experience or 'no' in experience:
                profile['Trainability Level'] = 5
                profile['Adaptability Level'] = 5
                weights['Trainability Level'] = 2.5
                weights['Adaptability Level'] = 1.5
            elif 'experienced' in experience or 'yes' in experience or 'some' in experience:
                weights['Trainability Level'] = 1.0
        
        # Guard Dog
        if 'guard_dog' in preferences:
            guard = preferences['guard_dog'].lower()
            if 'yes' in guard:
                profile['Watchdog/Protective Nature'] = 5
                weights['Watchdog/Protective Nature'] = 2.0
            elif 'no' in guard:
                profile['Openness To Strangers'] = 5
                weights['Openness To Strangers'] = 1.5
        
        # Barking Tolerance
        if 'barking_tolerance' in preferences:
            barking = preferences['barking_tolerance'].lower()
            if 'quiet' in barking or 'low' in barking:
                profile['Barking Level'] = 1
                weights['Barking Level'] = 2.0
            elif 'moderate' in barking:
                weights['Barking Level'] = 0.5
        
        # Set default weights for traits not specified
        for trait in self.trait_columns:
            if trait not in weights:
                weights[trait] = 0.3
        
        return profile, weights
    
    def calculate_match_score(self, breed_row, user_profile, weights):
        """Calculate weighted match score for a breed."""
        score = 0
        max_score = 0
        
        for trait in self.trait_columns:
            weight = weights.get(trait, 0.3)
            max_score += weight * 5
            
            if trait in user_profile:
                diff = abs(breed_row[trait] - user_profile[trait])
                trait_score = (5 - diff) * weight
                score += trait_score
            else:
                score += 3 * weight
        
        return (score / max_score) * 100
    
    def apply_dealbreakers(self, breeds_df, preferences):
        """Filter out breeds that don't meet critical requirements."""
        filtered = breeds_df.copy()
        
        # Allergies - must be low shedding
        if 'allergies' in preferences and 'yes' in preferences['allergies'].lower():
            filtered = filtered[filtered['Shedding Level'] <= 2]
        
        # Young children - must be at least moderately good
        if 'children' in preferences and 'yes' in preferences['children'].lower():
            filtered = filtered[filtered['Good With Young Children'] >= 3]
        
        # Other dogs - must be compatible
        if 'other_dogs' in preferences and 'yes' in preferences['other_dogs'].lower():
            filtered = filtered[filtered['Good With Other Dogs'] >= 3]
        
        return filtered
    
    def get_recommendations(self, preferences, top_n=3):
        """Get top N breed recommendations with explanations."""
        user_profile, weights = self.create_user_profile(preferences)
        
        # Apply dealbreakers on original data
        eligible_breeds_original = self.apply_dealbreakers(self.breed_traits_original, preferences)
        
        if len(eligible_breeds_original) == 0:
            return []
        
        # Get corresponding normalized breeds
        eligible_breed_names = eligible_breeds_original['Breed'].tolist()
        eligible_breeds = self.breed_traits[self.breed_traits['Breed'].isin(eligible_breed_names)]
        
        # Calculate match scores
        scores = []
        for idx, row in eligible_breeds.iterrows():
            score = self.calculate_match_score(row, user_profile, weights)
            # Get original traits for display
            original_row = self.breed_traits_original[self.breed_traits_original['Breed'] == row['Breed']].iloc[0]
            scores.append({
                'breed': row['Breed'],
                'score': score,
                'traits': original_row.to_dict()
            })
        
        # Sort by score
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top N with explanations
        recommendations = []
        for i, match in enumerate(scores[:top_n]):
            explanation = self._generate_explanation(
                match['breed'], 
                match['traits'], 
                preferences, 
                user_profile
            )
            
            recommendations.append({
                'rank': i + 1,
                'breed': match['breed'],
                'score': round(match['score'], 1),
                'explanation': explanation,
                'traits': match['traits']
            })
        
        return recommendations
    
    def _generate_explanation(self, breed_name, traits, preferences, user_profile):
        """Generate natural language explanation for the match."""
        reasons = []
        
        # Check key matching factors
        if 'children' in preferences and 'yes' in preferences['children'].lower():
            child_score = traits['Good With Young Children']
            if child_score >= 4:
                reasons.append(f"excellent with children (rated {child_score}/5)")
        
        if 'activity_level' in preferences:
            energy = traits['Energy Level']
            if 'high' in preferences['activity_level'].lower() or 'very active' in preferences['activity_level'].lower():
                if energy >= 4:
                    reasons.append(f"high energy level ({energy}/5) matches your active lifestyle")
            elif 'low' in preferences['activity_level'].lower() or 'calm' in preferences['activity_level'].lower():
                if energy <= 3:
                    reasons.append(f"calm demeanor (energy {energy}/5) suits a relaxed home")
            elif 'moderate' in preferences['activity_level'].lower():
                if 2 <= energy <= 4:
                    reasons.append(f"moderate energy level ({energy}/5) fits your lifestyle")
        
        if 'allergies' in preferences and 'yes' in preferences['allergies'].lower():
            shedding = traits['Shedding Level']
            if shedding <= 2:
                reasons.append(f"low shedding ({shedding}/5) - great for allergies")
        
        if 'training_experience' in preferences:
            if 'beginner' in preferences['training_experience'].lower() or 'first' in preferences['training_experience'].lower():
                trainability = traits['Trainability Level']
                if trainability >= 4:
                    reasons.append(f"highly trainable ({trainability}/5) - perfect for first-time owners")
        
        # Add affection note
        affection = traits['Affectionate With Family']
        if affection >= 4:
            reasons.append(f"very affectionate ({affection}/5)")
        
        if not reasons:
            reasons.append("well-balanced traits for your lifestyle")
        
        return f"{breed_name} is recommended because it's " + ", ".join(reasons) + "."

# Load data
@st.cache_data
def load_data():
    breed_traits = pd.read_csv('data/breed_traits.csv')
    breed_traits['Breed'] = breed_traits['Breed'].str.replace('Â', ' ').str.strip()
    return breed_traits

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'preferences' not in st.session_state:
    st.session_state.preferences = {}
if 'stage' not in st.session_state:
    st.session_state.stage = 'greeting'
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Load data
breed_traits = load_data()
matchmaker = DogMatchmaker(breed_traits)

# Header
st.markdown('<p class="main-header">🐕 Dog Matchmaker AI 🐾</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Personal AI Companion for Finding the Perfect Dog Breed</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI-powered chatbot helps you find the perfect dog breed based on your lifestyle, 
    preferences, and living situation.
    
    **How it works:**
    1. Answer questions about your lifestyle
    2. Get personalized breed recommendations
    3. View detailed profiles with images
    """)
    
    st.header("📊 Quick Stats")
    st.metric("Total Breeds", len(breed_traits))
    st.metric("Traits Analyzed", 16)
    
    if st.button("🔄 Start Over"):
        st.session_state.conversation = []
        st.session_state.preferences = {}
        st.session_state.stage = 'greeting'
        st.session_state.recommendations = None
        st.rerun()

# Main chat interface
st.header("💬 Chat with the Dog Matchmaker")

# Display conversation history
for message in st.session_state.conversation:
    if message['role'] == 'user':
        st.info(f"👤 **You:** {message['content']}")
    else:
        st.success(f"🤖 **Dog Matchmaker:** {message['content']}")

# Greeting
if st.session_state.stage == 'greeting' and len(st.session_state.conversation) == 0:
    greeting = """
    Hello! 🐕 I'm your Dog Matchmaker AI, and I'm here to help you find the perfect dog breed 
    for your lifestyle! 
    
    I'll ask you a few questions about your living situation, activity level, and preferences. 
    Based on your answers, I'll recommend the top 3 dog breeds that match you best.
    
    Let's get started! What's your activity level like? Are you very active, moderately active, 
    or do you prefer calm, relaxed activities?
    """
    st.session_state.conversation.append({'role': 'bot', 'content': greeting})
    st.session_state.stage = 'activity_level'
    st.rerun()

# Question flow
questions = {
    'activity_level': "What's your activity level like? (e.g., very active, moderate, prefer calm activities)",
    'living_space': "What's your living situation? (e.g., apartment, house with yard, house without yard)",
    'children': "Do you have young children at home? (yes/no)",
    'other_dogs': "Do you have other dogs at home? (yes/no)",
    'allergies': "Do you or anyone in your household have pet allergies? (yes/no)",
    'grooming_time': "How much time can you dedicate to grooming? (minimal, moderate, lots of time)",
    'training_experience': "Have you owned dogs before? (first-time owner, some experience, very experienced)",
    'guard_dog': "Are you looking for a protective/guard dog? (yes/no)",
    'barking_tolerance': "How do you feel about barking? (prefer quiet, don't mind, no preference)"
}

question_order = list(questions.keys())

# User input with form to prevent auto-fill
if st.session_state.stage not in ['results', 'processing']:
    with st.form(key=f"input_form_{st.session_state.stage}", clear_on_submit=True):
        user_input = st.text_input(
            "Your answer:", 
            placeholder="Type your answer here...",
            key=f"text_input_{st.session_state.stage}"
        )
        submit = st.form_submit_button("Send ➤")
        
        if submit and user_input.strip():
            # Add user message
            st.session_state.conversation.append({'role': 'user', 'content': user_input})
            
            # Store preference
            st.session_state.preferences[st.session_state.stage] = user_input
            
            # Move to next question
            if st.session_state.stage in question_order:
                current_index = question_order.index(st.session_state.stage)
                
                if current_index < len(question_order) - 1:
                    next_stage = question_order[current_index + 1]
                    st.session_state.stage = next_stage
                    
                    bot_response = f"Got it! {questions[next_stage]}"
                    st.session_state.conversation.append({'role': 'bot', 'content': bot_response})
                else:
                    st.session_state.stage = 'processing'
                    bot_response = "Perfect! I have all the information I need. Let me find the best breeds for you... 🔍"
                    st.session_state.conversation.append({'role': 'bot', 'content': bot_response})
            
            st.rerun()

# Generate recommendations
if st.session_state.stage == 'processing':
    with st.spinner("Analyzing breed compatibility..."):
        recommendations = matchmaker.get_recommendations(st.session_state.preferences, top_n=3)
        st.session_state.recommendations = recommendations
        st.session_state.stage = 'results'
    
    if recommendations:
        bot_response = f"Great news! Based on your preferences, I've found {len(recommendations)} perfect breed matches for you!"
    else:
        bot_response = "I'm having trouble finding breeds that match all your requirements. Let me show you some close matches instead!"
        # Get recommendations without dealbreakers
        recommendations = matchmaker.get_recommendations({}, top_n=3)
        st.session_state.recommendations = recommendations
    
    st.session_state.conversation.append({'role': 'bot', 'content': bot_response})
    st.rerun()

# Display recommendations
if st.session_state.stage == 'results' and st.session_state.recommendations:
    st.markdown("---")
    st.header("🏆 Your Top 3 Breed Matches")
    
    for rec in st.session_state.recommendations:
        with st.expander(f"#{rec['rank']} {rec['breed']} - Match Score: {rec['score']}%", expanded=True):
            
            # Create two columns: image and details
            col_img, col_details = st.columns([1, 2])
            
            with col_img:
                # Try to load and display image
                img_path = get_image_path(rec['breed'])
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_column_width=True, caption=rec['breed'])
                else:
                    st.info(f"📷 Image not available")
            
            with col_details:
                st.write(rec['explanation'])
                
                st.markdown("**Key Traits:**")
                trait_col1, trait_col2, trait_col3 = st.columns(3)
                
                with trait_col1:
                    st.metric("Energy Level", f"{int(rec['traits']['Energy Level'])}/5", "⚡")
                    st.metric("Trainability", f"{int(rec['traits']['Trainability Level'])}/5", "🎓")
                
                with trait_col2:
                    st.metric("Good With Children", f"{int(rec['traits']['Good With Young Children'])}/5", "👶")
                    st.metric("Shedding Level", f"{int(rec['traits']['Shedding Level'])}/5", "🧹")
                
                with trait_col3:
                    st.metric("Affection", f"{int(rec['traits']['Affectionate With Family'])}/5", "❤️")
                    st.metric("Grooming Needs", f"{int(rec['traits']['Coat Grooming Frequency'])}/5", "✂️")
    
    # Social media post
    st.markdown("---")
    st.header("📱 Share Your Match")
    
    top_match = st.session_state.recommendations[0]
    social_post = f"""🐕 I just found my perfect dog match! 🎉

According to the Dog Matchmaker AI, {top_match['breed']} is my #1 match with a {top_match['score']}% compatibility score!

{top_match['explanation']}

Find your perfect breed at Dog Matchmaker AI! 🐾
#DogMatchmaker #PerfectPup #DogLovers"""
    
    st.text_area("Copy this post to share on social media:", social_post, height=200)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐕 Dog Matchmaker AI | Powered by DataCamp Competition 2024</p>
    <p>Data source: Dog Breeds Dataset with 16 behavioral and physical traits</p>
</div>
""", unsafe_allow_html=True)