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
import base64
from pathlib import Path

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
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
    }
    .bot-message {
        background-color: #F1F8E9;
        border-left: 5px solid #8BC34A;
    }
    .breed-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 2px solid #FF6B6B;
    }
    .score-badge {
        background-color: #4ECDC4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .trait-pill {
        background-color: #FFE5B4;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #FF5252;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Dog Matchmaker Class
class DogMatchmaker:
    def __init__(self, breed_traits_df):
        self.breed_traits_original = breed_traits_df.copy()  # Keep original for display
        self.breed_traits = breed_traits_df.copy()
        
        # Get only numeric trait columns (exclude 'Breed', 'Coat Type', 'Coat Length')
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
            elif 'low' in activity or 'couch' in activity:
                profile['Energy Level'] = 2
                weights['Energy Level'] = 2.0
        
        # Living Space
        if 'living_space' in preferences:
            space = preferences['living_space'].lower()
            if 'apartment' in space or 'small' in space:
                profile['Adaptability Level'] = 5
                weights['Adaptability Level'] = 2.0
        
        # Children
        if 'children' in preferences:
            children = preferences['children'].lower()
            if 'yes' in children:
                profile['Good With Young Children'] = 5
                profile['Affectionate With Family'] = 5
                weights['Good With Young Children'] = 3.0
            else:
                weights['Good With Young Children'] = 0.1
        
        # Other Dogs
        if 'other_dogs' in preferences:
            other_dogs = preferences['other_dogs'].lower()
            if 'yes' in other_dogs:
                profile['Good With Other Dogs'] = 5
                weights['Good With Other Dogs'] = 3.0
            else:
                weights['Good With Other Dogs'] = 0.1
        
        # Allergies
        if 'allergies' in preferences:
            allergies = preferences['allergies'].lower()
            if 'yes' in allergies:
                profile['Shedding Level'] = 1
                weights['Shedding Level'] = 3.0
        
        # Grooming
        if 'grooming_time' in preferences:
            grooming = preferences['grooming_time'].lower()
            if 'low' in grooming or 'minimal' in grooming:
                profile['Coat Grooming Frequency'] = 1
                weights['Coat Grooming Frequency'] = 2.0
        
        # Training Experience
        if 'training_experience' in preferences:
            experience = preferences['training_experience'].lower()
            if 'first' in experience or 'beginner' in experience or 'no' in experience:
                profile['Trainability Level'] = 5
                weights['Trainability Level'] = 2.5
        
        # Guard Dog
        if 'guard_dog' in preferences:
            guard = preferences['guard_dog'].lower()
            if 'yes' in guard:
                profile['Watchdog/Protective Nature'] = 5
                weights['Watchdog/Protective Nature'] = 2.0
            else:
                profile['Openness To Strangers'] = 5
                weights['Openness To Strangers'] = 1.5
        
        # Barking
        if 'barking_tolerance' in preferences:
            barking = preferences['barking_tolerance'].lower()
            if 'quiet' in barking or 'low' in barking:
                profile['Barking Level'] = 1
                weights['Barking Level'] = 2.0
        
        # Set defaults
        for trait in self.trait_columns:
            if trait not in weights:
                weights[trait] = 0.3
        
        return profile, weights
    
    def calculate_match_score(self, breed_row, user_profile, weights):
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
        filtered = breeds_df.copy()
        
        if 'allergies' in preferences and 'yes' in preferences['allergies'].lower():
            filtered = filtered[filtered['Shedding Level'] <= 2]
        
        if 'children' in preferences and 'yes' in preferences['children'].lower():
            filtered = filtered[filtered['Good With Young Children'] >= 3]
        
        if 'other_dogs' in preferences and 'yes' in preferences['other_dogs'].lower():
            filtered = filtered[filtered['Good With Other Dogs'] >= 3]
        
        return filtered
    
    def get_recommendations(self, preferences, top_n=3):
        user_profile, weights = self.create_user_profile(preferences)
        
        # Apply dealbreakers on original data
        eligible_breeds_original = self.apply_dealbreakers(self.breed_traits_original, preferences)
        
        if len(eligible_breeds_original) == 0:
            return []
        
        # Get corresponding normalized breeds
        eligible_breed_names = eligible_breeds_original['Breed'].tolist()
        eligible_breeds = self.breed_traits[self.breed_traits['Breed'].isin(eligible_breed_names)]
        
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
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        recommendations = []
        for i, match in enumerate(scores[:top_n]):
            explanation = self._generate_explanation(match['breed'], match['traits'], preferences)
            
            recommendations.append({
                'rank': i + 1,
                'breed': match['breed'],
                'score': round(match['score'], 1),
                'explanation': explanation,
                'traits': match['traits']
            })
        
        return recommendations
    
    def _generate_explanation(self, breed_name, traits, preferences):
        reasons = []
        
        if 'children' in preferences and 'yes' in preferences['children'].lower():
            if traits['Good With Young Children'] >= 4:
                reasons.append(f"excellent with children (rated {int(traits['Good With Young Children'])}/5)")
        
        if 'activity_level' in preferences:
            energy = traits['Energy Level']
            if 'high' in preferences['activity_level'].lower() and energy >= 4:
                reasons.append(f"high energy level matches your active lifestyle")
            elif 'low' in preferences['activity_level'].lower() and energy <= 3:
                reasons.append(f"calm demeanor suits a relaxed home")
        
        if 'allergies' in preferences and 'yes' in preferences['allergies'].lower():
            if traits['Shedding Level'] <= 2:
                reasons.append(f"low shedding - great for allergies")
        
        if 'training_experience' in preferences and ('beginner' in preferences['training_experience'].lower() or 'no' in preferences['training_experience'].lower()):
            if traits['Trainability Level'] >= 4:
                reasons.append(f"highly trainable - perfect for first-time owners")
        
        if traits['Affectionate With Family'] >= 4:
            reasons.append(f"very affectionate with family")
        
        if not reasons:
            reasons.append("well-balanced traits for your lifestyle")
        
        return f"{breed_name} is recommended because it's " + ", ".join(reasons) + "."

# Load data
@st.cache_data
def load_data():
    breed_traits = pd.read_csv('data/breed_traits.csv')
    breed_traits['Breed'] = breed_traits['Breed'].str.replace('Â', ' ').str.strip()
    return breed_traits

# Get image path
def get_image_path(breed_name):
    """Get the image path for a breed from the sample images folder."""
    # Clean breed name for filename
    clean_name = breed_name.replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')
    
    # Common image locations to try in the sample folder
    possible_paths = [
        f'Dog-Breeds-Dataset/Images-Sample/{clean_name}.jpg',
        f'Dog-Breeds-Dataset/Images-Sample/{clean_name}.png',
        f'Dog-Breeds-Dataset/Images-Sample/{clean_name}.jpeg',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

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
    1. Answer a few questions about your lifestyle
    2. Get personalized breed recommendations
    3. View detailed profiles with images
    4. Make an informed decision!
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
        st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">🤖 <strong>Dog Matchmaker:</strong> {message["content"]}</div>', unsafe_allow_html=True)

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

# User input
if st.session_state.stage != 'results':
    user_input = st.text_input("Your answer:", key="user_input", placeholder="Type your answer here...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Send", type="primary")
    
    if submit and user_input:
        # Add user message
        st.session_state.conversation.append({'role': 'user', 'content': user_input})
        
        # Store preference
        st.session_state.preferences[st.session_state.stage] = user_input
        
        # Move to next question
        current_index = question_order.index(st.session_state.stage)
        
        if current_index < len(question_order) - 1:
            next_stage = question_order[current_index + 1]
            st.session_state.stage = next_stage
            
            # Add bot response with next question
            bot_response = f"Got it! {questions[next_stage]}"
            st.session_state.conversation.append({'role': 'bot', 'content': bot_response})
        else:
            # All questions answered, generate recommendations
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
        bot_response = f"""
        Great news! Based on your preferences, I've found {len(recommendations)} perfect breed matches for you! 
        Here are your personalized recommendations:
        """
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
        with st.container():
            st.markdown(f'<div class="breed-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Try to load image
                img_path = get_image_path(rec['breed'])
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                else:
                    st.info(f"📷 Image for {rec['breed']} not found")
            
            with col2:
                st.markdown(f"### #{rec['rank']} {rec['breed']}")
                st.markdown(f'<div class="score-badge">Match Score: {rec["score"]}%</div>', unsafe_allow_html=True)
                st.write(rec['explanation'])
                
                # Show key traits
                st.markdown("**Key Traits:**")
                trait_cols = st.columns(3)
                
                key_traits = [
                    ('Energy Level', rec['traits']['Energy Level']),
                    ('Trainability', rec['traits']['Trainability Level']),
                    ('Good With Children', rec['traits']['Good With Young Children']),
                    ('Shedding', rec['traits']['Shedding Level']),
                    ('Affection', rec['traits']['Affectionate With Family']),
                    ('Grooming Needs', rec['traits']['Coat Grooming Frequency'])
                ]
                
                for idx, (trait, value) in enumerate(key_traits):
                    with trait_cols[idx % 3]:
                        st.markdown(f'<div class="trait-pill">{trait}: {"⭐" * int(value)}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Social media post generation
    st.markdown("---")
    st.header("📱 Share Your Match")
    
    top_match = st.session_state.recommendations[0]
    social_post = f"""
    🐕 I just found my perfect dog match! 🎉
    
    According to the Dog Matchmaker AI, {top_match['breed']} is my #1 match with a {top_match['score']}% compatibility score!
    
    {top_match['explanation']}
    
    Find your perfect breed at Dog Matchmaker AI! 🐾
    #DogMatchmaker #PerfectPup #DogLovers
    """
    
    st.text_area("Copy this post to share on social media:", social_post, height=200)
    
    # Interactive Comparison Chart
    st.markdown("---")
    st.header("📊 Interactive Breed Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for rec in st.session_state.recommendations:
        comparison_data.append({
            'Breed': rec['breed'],
            'Match Score': rec['score'],
            'Energy Level': int(rec['traits']['Energy Level']),
            'Trainability': int(rec['traits']['Trainability Level']),
            'Child Friendly': int(rec['traits']['Good With Young Children']),
            'Shedding': int(rec['traits']['Shedding Level']),
            'Affection': int(rec['traits']['Affectionate With Family']),
            'Adaptability': int(rec['traits']['Adaptability Level'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Bar chart comparison
    traits_to_compare = ['Energy Level', 'Trainability', 'Child Friendly', 'Affection', 'Adaptability']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(traits_to_compare))
    width = 0.25
    
    for i, rec in enumerate(st.session_state.recommendations):
        values = [comparison_df.iloc[i][trait] for trait in traits_to_compare]
        ax.bar(x + i*width, values, width, label=f"#{i+1} {rec['breed']}", alpha=0.8)
    
    ax.set_xlabel('Traits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rating (1-5)', fontsize=12, fontweight='bold')
    ax.set_title('Side-by-Side Trait Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(traits_to_compare, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed comparison table
    st.markdown("### Detailed Trait Comparison")
    
    # Format the dataframe for display
    display_df = comparison_df.copy()
    display_df['Match Score'] = display_df['Match Score'].apply(lambda x: f"{x}%")
    
    # Add star ratings
    for col in ['Energy Level', 'Trainability', 'Child Friendly', 'Shedding', 'Affection', 'Adaptability']:
        display_df[col] = display_df[col].apply(lambda x: '⭐' * x)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button for results
    results_text = f"""
YOUR DOG BREED RECOMMENDATIONS
================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

YOUR PREFERENCES:
{'-' * 40}
"""
    for key, value in st.session_state.preferences.items():
        results_text += f"{key.replace('_', ' ').title()}: {value}\n"
    
    results_text += f"\n\nTOP 3 BREED MATCHES:\n{'=' * 40}\n\n"
    
    for rec in st.session_state.recommendations:
        results_text += f"#{rec['rank']} {rec['breed']} - Match Score: {rec['score']}%\n"
        results_text += f"{rec['explanation']}\n\n"
        results_text += "Key Traits:\n"
        results_text += f"  - Energy Level: {int(rec['traits']['Energy Level'])}/5\n"
        results_text += f"  - Trainability: {int(rec['traits']['Trainability Level'])}/5\n"
        results_text += f"  - Good With Children: {int(rec['traits']['Good With Young Children'])}/5\n"
        results_text += f"  - Shedding Level: {int(rec['traits']['Shedding Level'])}/5\n\n"
    
    st.download_button(
        label="📥 Download Your Results",
        data=results_text,
        file_name="dog_matchmaker_results.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐕 Dog Matchmaker AI | Powered by DataCamp Competition 2024</p>
    <p>Data source: Dog Breeds Dataset with 16 behavioral and physical traits</p>
</div>
""", unsafe_allow_html=True)
