# dashboard/app.py
# Plotly Dash chat interface for the RAG pipeline

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from loguru import logger
from config.settings import get_settings

settings = get_settings()
API_URL = f"http://localhost:{settings.api_port}"

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🤖 AI-Powered Data Pipeline",
                    className="text-center mt-4 mb-1"),
            html.P("RAG System — LangChain + ChromaDB + Llama 3.2 (Local, Free)",
                   className="text-center text-muted mb-4")
        ])
    ]),

    # Stats Row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("0", id="chunk-count", className="text-info"),
                html.P("Chunks in Vector DB", className="text-muted mb-0")
            ])
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Llama 3.2 3B", className="text-success"),
                html.P("Local LLM (Free)", className="text-muted mb-0")
            ])
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("ChromaDB", className="text-warning"),
                html.P("Vector Database", className="text-muted mb-0")
            ])
        ]), width=4),
    ], className="mb-4"),

    # Ingest Button
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "🔄 Ingest Data (Wikipedia + ArXiv)",
                id="ingest-btn",
                color="primary",
                className="w-100 mb-2"
            ),
            html.Div(id="ingest-status", className="text-center text-muted")
        ])
    ], className="mb-4"),

    # Chat Interface
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("💬 Ask a Question"),
                dbc.CardBody([
                    # Chat history
                    html.Div(
                        id="chat-history",
                        style={
                            "height": "400px",
                            "overflowY": "auto",
                            "padding": "10px"
                        }
                    ),
                    html.Hr(),
                    # Input
                    dbc.InputGroup([
                        dbc.Input(
                            id="question-input",
                            placeholder="Ask about Data Engineering, Kafka, RAG...",
                            type="text",
                            debounce=False
                        ),
                        dbc.Button(
                            "Ask",
                            id="ask-btn",
                            color="success",
                            n_clicks=0
                        )
                    ])
                ])
            ])
        ])
    ], className="mb-4"),

    # Sources Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📚 Sources Used"),
                dbc.CardBody(html.Div(id="sources-panel",
                             children=html.P("Ask a question to see sources",
                             className="text-muted")))
            ])
        ])
    ]),

    # Auto refresh stats
    dcc.Interval(id="stats-interval", interval=10000, n_intervals=0),

    # Store chat history
    dcc.Store(id="chat-store", data=[])

], fluid=True)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("chunk-count", "children"),
    Input("stats-interval", "n_intervals")
)
def update_stats(n):
    try:
        resp = requests.get(f"{API_URL}/stats", timeout=3)
        data = resp.json()
        return str(data.get("total_chunks", 0))
    except:
        return "API offline"

@callback(
    Output("ingest-status", "children"),
    Input("ingest-btn", "n_clicks"),
    prevent_initial_call=True
)
def trigger_ingest(n_clicks):
    try:
        resp = requests.post(f"{API_URL}/ingest", timeout=120)
        data = resp.json()
        return f"✅ Ingested {data['documents_ingested']} docs, {data['chunks_stored']} chunks stored"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@callback(
    Output("chat-history", "children"),
    Output("sources-panel", "children"),
    Output("chat-store", "data"),
    Output("question-input", "value"),
    Input("ask-btn", "n_clicks"),
    Input("question-input", "n_submit"),
    State("question-input", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True
)
def ask_question(n_clicks, n_submit, question, chat_history):
    if not question or not question.strip():
        return chat_history_to_html(chat_history), no_sources(), chat_history, ""

    try:
        resp = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": 3},
            timeout=120
        )
        data = resp.json()
        answer  = data.get("answer", "No answer")
        sources = data.get("sources", [])

        # Add to chat history
        chat_history.append({"question": question, "answer": answer})

        # Build sources display
        sources_display = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Title"),
                html.Th("Source"),
                html.Th("Relevance")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(html.A(s["title"], href=s.get("url","#"),
                            target="_blank")),
                    html.Td(s["source"]),
                    html.Td(f"{(1 - s['distance']) * 100:.0f}%")
                ]) for s in sources
            ])
        ], striped=True, hover=True, size="sm")

        return chat_history_to_html(chat_history), sources_display, chat_history, ""

    except Exception as e:
        chat_history.append({"question": question, "answer": f"Error: {str(e)}"})
        return chat_history_to_html(chat_history), no_sources(), chat_history, ""

def chat_history_to_html(history):
    if not history:
        return html.P("No questions yet. Ask something!", className="text-muted")
    messages = []
    for item in history:
        messages.append(
            html.Div([
                html.Div([
                    html.Strong("You: "),
                    html.Span(item["question"])
                ], style={"background": "#1a2030", "padding": "8px",
                          "borderRadius": "8px", "marginBottom": "5px"}),
                html.Div([
                    html.Strong("AI: "),
                    html.Span(item["answer"])
                ], style={"background": "#0d2818", "padding": "8px",
                          "borderRadius": "8px", "marginBottom": "15px"})
            ])
        )
    return messages

def no_sources():
    return html.P("No sources yet", className="text-muted")


if __name__ == "__main__":
    logger.info(f"Dashboard starting at http://localhost:{settings.dash_port}")
    app.run(debug=True, host="0.0.0.0", port=settings.dash_port)