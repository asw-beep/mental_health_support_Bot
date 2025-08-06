import gradio as gr
import openai
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import os
from typing import Dict, List
import json
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key (will now load from .env file)
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
    print("Make sure your .env file contains: OPENAI_API_KEY=your-api-key-here")

class MoodAnalysisTool(BaseTool):
    name: str = "mood_analyzer"
    description: str = "Analyzes text to detect emotional state and mood patterns"
    
    def _run(self, text: str) -> str:
        """Analyze mood from text input"""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are an expert mood analyzer. Analyze the given text and provide:
                    1. Primary emotion (happy, sad, anxious, angry, neutral, excited, overwhelmed, etc.)
                    2. Intensity level (1-10)
                    3. Key emotional indicators found in the text
                    4. Overall mood assessment
                    
                    Return as JSON format with keys: primary_emotion, intensity, indicators, assessment"""},
                    {"role": "user", "content": f"Analyze the mood in this text: {text}"}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing mood: {str(e)}"

class SelfCareRecommendationTool(BaseTool):
    name: str = "selfcare_recommender"
    description: str = "Provides personalized self-care recommendations based on mood and preferences"
    
    def _run(self, mood_analysis: str, user_preferences: str = "") -> str:
        """Generate self-care recommendations"""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a certified wellness coach. Based on the mood analysis provided, 
                    suggest 3-5 specific, actionable self-care activities. Include:
                    1. Immediate actions (can be done right now)
                    2. Short-term activities (within the day)
                    3. Longer-term wellness practices
                    4. Emergency resources if needed
                    
                    Make recommendations practical, evidence-based, and supportive."""},
                    {"role": "user", "content": f"Mood analysis: {mood_analysis}\nUser preferences: {user_preferences}"}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

class CompanionChatTool(BaseTool):
    name: str = "companion_chat"
    description: str = "Provides empathetic conversation and emotional support"
    
    def _run(self, user_message: str, mood_context: str = "") -> str:
        """Provide compassionate response"""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a compassionate AI companion focused on mental wellness. 
                    Provide empathetic, supportive responses that:
                    1. Validate the user's feelings
                    2. Offer gentle encouragement
                    3. Ask thoughtful follow-up questions when appropriate
                    4. Maintain professional boundaries while being warm
                    5. Suggest professional help if serious concerns arise
                    
                    Be genuine, non-judgmental, and supportive."""},
                    {"role": "user", "content": f"User message: {user_message}\nMood context: {mood_context}"}
                ],
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in companion chat: {str(e)}"

# Initialize tools
mood_tool = MoodAnalysisTool()
selfcare_tool = SelfCareRecommendationTool()
companion_tool = CompanionChatTool()

# Create Agents
mood_analyzer = Agent(
    role='Mood Analyzer',
    goal='Accurately analyze emotional states and mood patterns from user text',
    backstory="""You are a specialized AI trained in emotional intelligence and psychology. 
    Your expertise lies in understanding subtle emotional cues in text and providing 
    accurate mood assessments that help other agents provide better support.""",
    tools=[mood_tool],
    verbose=True,
    allow_delegation=False
)

companion = Agent(
    role='Emotional Companion',
    goal='Provide empathetic support and meaningful conversation',
    backstory="""You are a warm, understanding companion who specializes in providing 
    emotional support. You listen actively, validate feelings, and offer comfort while 
    maintaining appropriate boundaries. Your responses are always kind, genuine, and supportive.""",
    tools=[companion_tool],
    verbose=True,
    allow_delegation=False
)

selfcare_recommender = Agent(
    role='Self-Care Specialist',
    goal='Recommend personalized wellness activities and coping strategies',
    backstory="""You are a certified wellness coach with expertise in mental health 
    self-care practices. You provide evidence-based, practical recommendations tailored 
    to individual needs and emotional states.""",
    tools=[selfcare_tool],
    verbose=True,
    allow_delegation=False
)

class MentalHealthSupportBot:
    def __init__(self):
        self.conversation_history = []
        
    def analyze_and_support(self, user_input: str, chat_history: List, preferences: str = ""):
        """Main function that coordinates all agents"""
        
        # Task 1: Analyze mood
        mood_task = Task(
            description=f"Analyze the emotional state and mood from this user input: '{user_input}'",
            agent=mood_analyzer,
            expected_output="JSON format mood analysis with primary emotion, intensity, indicators, and assessment"
        )
        
        # Task 2: Provide companion response
        companion_task = Task(
            description=f"Provide an empathetic, supportive response to: '{user_input}'",
            agent=companion,
            expected_output="Warm, supportive response that validates feelings and offers encouragement"
        )
        
        # Task 3: Generate self-care recommendations
        selfcare_task = Task(
            description=f"Based on the mood analysis, provide personalized self-care recommendations for: '{user_input}'. User preferences: {preferences}",
            agent=selfcare_recommender,
            expected_output="3-5 specific, actionable self-care recommendations categorized by timeframe"
        )
        
        # Create and run crew
        crew = Crew(
            agents=[mood_analyzer, companion, selfcare_recommender],
            tasks=[mood_task, companion_task, selfcare_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # Execute the crew
            result = crew.kickoff()
            
            # Parse results
            mood_analysis = mood_task.output.raw if hasattr(mood_task, 'output') else "Mood analysis completed"
            companion_response = companion_task.output.raw if hasattr(companion_task, 'output') else "Support provided"
            selfcare_recommendations = selfcare_task.output.raw if hasattr(selfcare_task, 'output') else "Recommendations generated"
            
            # Format response
            formatted_response = self._format_response(mood_analysis, companion_response, selfcare_recommendations)
            
            # Update chat history
            chat_history.append([user_input, formatted_response])
            
            return chat_history, ""
            
        except Exception as e:
            error_response = f"I'm here to support you, though I encountered a technical issue: {str(e)}. Please know that your feelings are valid and it's okay to seek help from friends, family, or mental health professionals."
            chat_history.append([user_input, error_response])
            return chat_history, ""
    
    def _format_response(self, mood_analysis: str, companion_response: str, selfcare_recommendations: str) -> str:
        """Format the combined response from all agents"""
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        response = f"""🌟 *Mental Health Support Response* - {timestamp}

💭 *Emotional Support:*
{companion_response}

🎯 *Mood Insights:*
{mood_analysis}

🌱 *Self-Care Recommendations:*
{selfcare_recommendations}

---
Remember: If you're experiencing severe distress or having thoughts of self-harm, please reach out to a mental health professional or crisis helpline immediately.

*Crisis Resources:*
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
"""
        return response

# Initialize the bot
support_bot = MentalHealthSupportBot()

# Create Gradio Interface
def create_interface():
    with gr.Blocks(title="Mental Health Support Bot", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🌈 Mental Health Support Bot
        
        Welcome to your personal mental health companion. Share how you're feeling, and I'll provide:
        - *Empathetic support* and understanding
        - *Mood analysis* to help you understand your emotions
        - *Personalized self-care recommendations*
        
        This bot is designed to complement, not replace, professional mental health care.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation",
                    show_label=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Share how you're feeling or what's on your mind...",
                        label="Your Message",
                        lines=3,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
                
            with gr.Column(scale=1):
                gr.Markdown("### Preferences & Settings")
                preferences = gr.Textbox(
                    label="Your Preferences",
                    placeholder="e.g., I prefer outdoor activities, I have limited time, I enjoy creative activities...",
                    lines=4
                )
                
                gr.Markdown("""
                ### Quick Self-Care Ideas
                - Take 5 deep breaths
                - Step outside for fresh air
                - Listen to calming music
                - Practice gratitude
                - Reach out to a friend
                - Take a short walk
                - Drink a glass of water
                """)
                
                gr.Markdown("""
                *If you need immediate help:*
                - 🇮🇳 National Mental Health Helpline: *1800-599-0019* (24/7 Toll-Free)
                - 🇮🇳 AASRA Suicide Prevention: *(+91) 9820466726*
                - 🇮🇳 Emergency Services: *112*
                - 🇮🇳 iCall Helpline: *+91 22-25521111* (Mon-Sat, 8 AM-10 PM)
                - 🌍 International: Contact your local emergency services
                """)
        
        # Event handlers
        def respond(message, history, prefs):
            if not message.strip():
                return history, ""
            return support_bot.analyze_and_support(message, history, prefs)
        
        send_btn.click(
            respond,
            inputs=[msg_input, chatbot, preferences],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, preferences],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg_input]
        )
        
        # Welcome message
        interface.load(
            lambda: [["", "Hello! I'm here to support you. How are you feeling today? Feel free to share anything that's on your mind. 💙"]],
            outputs=chatbot
        )
    
    return interface

# Launch the application
if __name__ == "__main__":
    # Check if API key is loaded from .env file
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: Please set your OPENAI_API_KEY environment variable")
        print("Create a .env file in your project directory with:")
        print("OPENAI_API_KEY=your-api-key-here")
        print("\nOr set it by running: export OPENAI_API_KEY='your-api-key-here'")
    else:
        print("✅ OpenAI API key loaded successfully from environment")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
