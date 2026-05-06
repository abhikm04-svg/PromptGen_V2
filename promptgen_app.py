"""
PromptGen V2 — AI-Powered Prompt Optimizer (Streamlit App)

This is the Streamlit UI entry point. All business logic lives in the `core/` package.
"""
import time
import io

try:
    import streamlit as st
except ImportError:
    raise ImportError("Streamlit is not installed. Run: pip install streamlit")

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from core.config import configure_client
from core.workflow import (
    initialize_workflow, get_clarification_questions, process_answers_and_optimize
)


def main():
    """Main Streamlit application"""
    # Page configuration — must be the first Streamlit command
    st.set_page_config(
        page_title="PromptGen V2 — AI Prompt Optimizer",
        page_icon="🧙",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Configure API key from Streamlit secrets
    if 'GOOGLE_API_KEY' in st.secrets:
        configure_client(st.secrets['GOOGLE_API_KEY'])
    else:
        st.error("⚠️ Please configure GOOGLE_API_KEY in Streamlit secrets")
        st.stop()

    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = initialize_workflow()
        st.session_state.stage = 'idea'
        st.session_state.questions = []
        st.session_state.results = None
        st.session_state.token_data = []

    # Stage 1: Get user idea
    if st.session_state.stage == 'idea':
        st.title("🧙 Prompt Optimizer")
        st.markdown("Welcome! I'm Skadoosh! Your very own prompt wizard. Let's create the perfect prompt for your usecase")

        user_idea = st.text_area(
            'Enter your prompt idea:',
            placeholder="e.g., I want to create a prompt for writing engaging blog posts about technology...",
            height=100
        )

        if st.button('Generate Clarification Questions', type='primary'):
            if user_idea.strip():
                with st.status("Thinking... Generating clarification questions", state="running") as status:
                    try:
                        questions = get_clarification_questions(
                            st.session_state.workflow, user_idea
                        )
                        st.session_state.questions = questions
                        st.session_state.stage = 'questions'
                        status.update(label="Questions generated!", state="complete")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        status.update(label="Error generating questions!", state="error")
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a prompt idea first.")

    # Stage 2: Answer clarification questions
    elif st.session_state.stage == 'questions':
        st.title("📝 Answer Clarification Questions")
        st.markdown("Please answer the following questions to help me create the perfect prompt:")

        answers = {}
        for i, question in enumerate(st.session_state.questions, 1):
            answer = st.text_input(
                f'**Question {i}:** {question}',
                key=f'answer_{i}',
                placeholder="Your answer here..."
            )
            if answer:
                answers[question] = answer

        col1, col2 = st.columns(2)
        with col1:
            if st.button('← Back to Idea'):
                st.session_state.stage = 'idea'
                st.rerun()

        with col2:
            if st.button('Submit Answers & Optimize →', type='primary'):
                if len(answers) == len(st.session_state.questions):
                    st.session_state.token_data = []

                    def token_callback(timestamp, tokens, stage):
                        st.session_state.token_data.append({
                            'timestamp': timestamp, 'tokens': tokens, 'stage': stage
                        })

                    try:
                        with st.status("Thinking... Starting optimization", state="running") as status:
                            def status_callback_for_spinner(status_text):
                                status.update(label=status_text)

                            results = process_answers_and_optimize(
                                st.session_state.workflow, answers,
                                stream_callback=None,
                                status_callback=status_callback_for_spinner,
                                token_callback=token_callback
                            )
                            st.session_state.results = results
                            st.session_state.stage = 'results'
                            status.update(label="Optimization complete!", state="complete")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error during optimization: {str(e)}")
                else:
                    st.warning("Please answer all questions before submitting.")

    # Stage 3: Display results
    elif st.session_state.stage == 'results':
        st.title("✨ Final Optimized Prompt")

        results = st.session_state.results

        # Display score and metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Score", f"{results['final_score']}/100")
        with col2:
            st.metric("Iterations", results['iterations'])
        with col3:
            st.metric("Status", "✓ Converged" if results['converged'] else "⚠ Max Iterations")

        # Display final prompt
        st.subheader("📄 Your Optimized Prompt:")
        st.code(results['final_prompt'], language='xml')

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Prompt",
                data=results['final_prompt'],
                file_name="optimized_prompt.xml",
                mime="text/xml"
            )

        with col2:
            if st.session_state.token_data and pd is not None and HAS_MATPLOTLIB:
                try:
                    df = pd.DataFrame(st.session_state.token_data)
                    df['cumulative'] = df['tokens'].cumsum()

                    if len(df) > 1:
                        df['time_elapsed'] = df['timestamp'] - df['timestamp'].iloc[0]

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df['time_elapsed'], df['cumulative'], linewidth=2, color='#1f77b4')
                        ax.fill_between(df['time_elapsed'], df['cumulative'], alpha=0.3, color='#1f77b4')
                        ax.set_xlabel('Time Elapsed (seconds)', fontsize=12)
                        ax.set_ylabel('Cumulative Tokens', fontsize=12)
                        ax.set_title('Token Usage Over Time', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        plt.close(fig)

                        st.download_button(
                            label="📊 Download Token Usage Graph",
                            data=buf, file_name="token_usage_graph.png", mime="image/png"
                        )
                    else:
                        st.info("📊 Token usage data available but insufficient for graph")
                except Exception as e:
                    st.warning(f"Could not generate token graph: {str(e)}")
            else:
                if not st.session_state.token_data:
                    st.info("📊 No token usage data available")
                elif not HAS_MATPLOTLIB:
                    st.info("📊 Install matplotlib to generate token usage graph")

        # Show iteration history
        with st.expander("📊 View Iteration History"):
            for hist in results['history']:
                st.markdown(f"**Iteration {hist['iteration']}** - Score: {hist['score']}/100")
                with st.expander(f"View details for iteration {hist['iteration']}"):
                    st.markdown("**Prompt:**")
                    st.code(hist['prompt'], language='xml')
                    st.markdown("**Output:**")
                    st.text(hist['output'][:500] + "..." if len(hist['output']) > 500 else hist['output'])
                    st.markdown("**Feedback:**")
                    st.text(hist['feedback'])

        # Start over button
        if st.button('🔄 Start Over', type='primary'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
