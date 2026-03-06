# Premium Web UI Templates for Agent Dashboard

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{agent_name} - Agent Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root[data-theme='dark'] {{
            --primary: #3b82f6; /* Electric Blue */
            --primary-hover: #2563eb;
            --bg: #0f172a;
            --sidebar-bg: #1e293b;
            --chat-bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.6);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.1);
            --accent: #10b981;
            --error: #ef4444;
            --input-bg: #1e293b;
            --user-bubble-bg: #3b82f6;
            --agent-bubble-bg: #1e293b;
            --user-text: #ffffff;
            --agent-text: #f8fafc;
        }}

        :root[data-theme='light'] {{
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg: #f8fafc;
            --sidebar-bg: #ffffff;
            --chat-bg: #f1f5f9;
            --card-bg: #ffffff;
            --text: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --accent: #10b981;
            --error: #ef4444;
            --input-bg: #ffffff;
            --user-bubble-bg: #2563eb;
            --agent-bubble-bg: #ffffff;
            --user-text: #ffffff;
            --agent-text: #0f172a;
        }}

        :root {{
            transition: all 0.2s ease;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}

        /* Sidebar Navigation */
        aside {{
            width: 280px;
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 1.25rem;
            flex-shrink: 0;
            z-index: 10;
        }}

        .agent-brand {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 2rem;
            padding: 0.5rem;
        }}

        .agent-emoji-small {{
            font-size: 1.75rem;
            background: rgba(99, 102, 241, 0.1);
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.15);
        }}

        .agent-brand-text h2 {{
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .agent-brand-text p {{
            font-size: 0.7rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 150px;
        }}

        nav {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            margin-bottom: 2rem;
        }}

        .nav-item {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.65rem 0.85rem;
            border-radius: 8px;
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            border: none;
            background: transparent;
            width: 100%;
            text-align: left;
            font-size: 0.88rem;
        }}

        .nav-item:hover {{
            background: rgba(255, 255, 255, 0.04);
            color: var(--text);
        }}

        .nav-item.active-primary {{
            background: var(--primary);
            color: white;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        }}

        /* Navbar Dropdown */
        .nav-dropdown {{
            position: relative;
        }}

        .nav-dropdown-content {{
            display: none;
            position: absolute;
            left: 100%;
            top: 0;
            background: var(--sidebar-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            min-width: 240px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
            z-index: 100;
            margin-left: 0.5rem;
            max-height: 400px;
            overflow-y: auto;
            padding: 0.5rem;
        }}

        .nav-dropdown:hover .nav-dropdown-content {{
            display: block;
        }}

        .theme-toggle {{
            margin-top: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: 10px;
            cursor: pointer;
            color: var(--text);
            font-size: 0.82rem;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}

        .theme-toggle:hover {{
            background: rgba(255, 255, 255, 0.08);
        }}

        /* History Section */
        .history-container {{
            flex-grow: 1;
            overflow-y: auto;
            margin: 0 -0.5rem;
            padding: 0 0.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .history-label {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 700;
            margin: 1rem 0 0.5rem 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .history-item {{
            padding: 0.55rem 0.75rem;
            border-radius: 6px;
            font-size: 0.82rem;
            color: var(--text-muted);
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.15s ease;
            max-width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .rename-btn {{
            opacity: 0;
            padding: 0.2rem;
            border-radius: 4px;
            transition: all 0.2s ease;
            cursor: pointer;
            font-size: 0.7rem;
        }}

        .history-item:hover .rename-btn {{
            opacity: 0.6;
        }}

        .rename-btn:hover {{
            opacity: 1 !important;
            background: rgba(255,255,255,0.1);
        }}

        .history-item:hover {{
            background: rgba(255, 255, 255, 0.03);
            color: var(--text);
        }}

        .history-item.active {{
            background: rgba(255, 255, 255, 0.05);
            color: var(--text);
            font-weight: 500;
            border-left: 2px solid var(--primary);
            border-radius: 0 6px 6px 0;
        }}

        .btn-new-chat {{
            margin-bottom: 1.5rem;
            justify-content: center;
            background: rgba(255, 255, 255, 0.05);
            border: 1px dashed var(--border);
            color: var(--text);
        }}

        .btn-new-chat:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--text-muted);
        }}

        /* Main Content */
        main {{
            flex-grow: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            background: var(--chat-bg);
            position: relative;
        }}

        .tab-content {{
            flex-grow: 1;
            display: none;
            overflow-y: auto;
            animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        #chat-tab {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}

        /* Native Chat Component */
        #chat-messages {{
            flex-grow: 1;
            overflow-y: auto;
            padding: 2.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            scroll-behavior: smooth;
        }}

        .message {{
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            animation: slideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .message-user {{
            align-items: flex-end;
        }}

        .message-assistant {{
            align-items: flex-start;
        }}

        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .message-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
            padding: 0 0.5rem;
        }}

        .message-user .message-header {{
            flex-direction: row-reverse;
        }}

        .message-avatar {{
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
        }}

        .message-assistant .message-avatar {{
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: var(--primary);
        }}

        .message-bubble {{
            max-width: 80%;
            padding: 0.85rem 1.25rem;
            border-radius: 18px;
            position: relative;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            line-height: 1.6;
            font-size: 0.98rem;
        }}

        .message-user .message-bubble {{
            background: var(--user-bubble-bg);
            color: var(--user-text);
            border-bottom-right-radius: 4px;
        }}

        .message-assistant .message-bubble {{
            background: var(--agent-bubble-bg);
            color: var(--agent-text);
            border-bottom-left-radius: 4px;
            border: 1px solid var(--border);
        }}

        .message-body {{
            width: 100%;
            display: flex;
            flex-direction: column;
        }}

        .message-user .message-body {{
            align-items: flex-end;
        }}

        .message-assistant .message-body {{
            align-items: flex-start;
        }}

        .message-content {{
            font-size: 0.98rem;
            line-height: 1.65;
        }}

        .message-content p {{
            margin-bottom: 1.1rem;
        }}

        .message-content p:last-child {{
            margin-bottom: 0;
        }}

        .message-content pre {{
            background: #111827;
            padding: 1.25rem;
            border-radius: 12px;
            overflow-x: auto;
            margin: 1.25rem 0;
            border: 1px solid var(--border);
        }}

        .message-content code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.2rem 0.45rem;
            border-radius: 5px;
            color: #d1d5db;
        }}

        .message-content pre code {{
            background: transparent;
            padding: 0;
            color: #e5e7eb;
        }}

        /* Reasoning / Thinking */
        .reasoning {{
            font-size: 0.88rem;
            color: var(--text-muted);
            border-left: 2px solid var(--border);
            padding-left: 1rem;
            margin-bottom: 1rem;
            font-style: italic;
            opacity: 0.85;
        }}

        /* Tool Calls */
        .tool-call {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 12px;
            margin: 1rem 0;
            overflow: hidden;
        }}

        .tool-header {{
            padding: 0.75rem 1.1rem;
            background: rgba(255, 255, 255, 0.015);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            font-size: 0.88rem;
            font-weight: 500;
        }}

        .tool-header:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}

        .tool-name {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: var(--text-muted);
        }}

        .tool-name strong {{
            color: var(--text);
        }}

        .tool-status {{
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .tool-status-running {{ color: var(--primary); }}
        .tool-status-done {{ color: var(--accent); }}

        .tool-body {{
            padding: 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            border-top: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.15);
        }}

        /* Chat Input */
        #chat-input-container {{
            padding: 0 2rem 2.5rem;
            max-width: 850px;
            width: 100%;
            margin: 0 auto;
        }}

        .input-wrapper {{
            background: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 0.85rem 1.25rem;
            display: flex;
            align-items: flex-end;
            gap: 1rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }}

        .input-wrapper:focus-within {{
            border-color: var(--primary);
        }}

        textarea#user-input {{
            flex-grow: 1;
            background: transparent;
            border: none;
            color: var(--text);
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            max-height: 250px;
            padding: 0.5rem 0;
            outline: none;
        }}

        .send-btn {{
            background: var(--primary);
            color: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            flex-shrink: 0;
            margin-bottom: 2px;
        }}

        .send-btn:hover:not(:disabled) {{
            background: var(--primary-hover);
        }}

        .send-btn:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}

        /* Management Layout */
        .content-inner {{
            padding: 3rem;
            max-width: 1200px;
            margin: 0 auto;
        }}

        .section-header {{
            margin-bottom: 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .section-header h2 {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .card {{
            background: var(--sidebar-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
        }}

        .btn {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.65rem 1.25rem;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            font-size: 0.88rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .btn-outline {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
        }}

        .list-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid var(--border);
        }}

        .badge {{
            background: rgba(16, 185, 129, 0.12);
            color: var(--accent);
            padding: 0.25rem 0.6rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        /* Modal */
        #editor-modal {{
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(8px);
        }}

        .modal-inner {{
            width: 90%;
            max-width: 1000px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2.25rem;
        }}

        textarea#editor-area {{
            width: 100%;
            height: 500px;
            background: #0b0f1a;
            color: #f1f5f9;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            margin: 1.5rem 0;
            resize: none;
        }}

        /* Mobile Header */
        .mobile-header {{
            display: none;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: var(--sidebar-bg);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 20;
        }}

        .hamburger {{
            background: transparent;
            border: none;
            color: var(--text);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        /* Overlay */
        .sidebar-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(4px);
            z-index: 15;
            animation: fadeIn 0.2s ease;
        }}

        /* Desktop vs Mobile adjustments */
        @media (max-width: 768px) {{
            aside {{
                position: fixed;
                left: -280px;
                top: 0;
                bottom: 0;
                z-index: 30;
                transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 10px 0 25px rgba(0, 0, 0, 0.2);
            }}

            aside.sidebar-open {{
                left: 0;
            }}

            .mobile-header {{
                display: flex;
            }}

            .message-bubble {{
                max-width: 90%;
            }}

            #chat-messages {{
                padding: 1.5rem 1rem;
            }}

            #chat-input-container {{
                padding: 0 1rem 1.5rem;
            }}

            .content-inner {{
                max-width: 1600px;
                margin: 0 auto;
                padding: 2rem;
            }}

            .grid {{
                grid-template-columns: 1fr;
            }}

            .sidebar-overlay.active {{
                display: block;
            }}
        }}
        /* Timeline & Schedule */
        .timeline-grid {{
            display: grid;
            grid-template-columns: 60px repeat(7, 1fr);
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow-x: auto;
            max-height: 600px;
        }}

        .timeline-header {{
            background: var(--sidebar-bg);
            padding: 0.75rem;
            font-weight: 700;
            font-size: 0.7rem;
            text-align: center;
            color: var(--text-muted);
            position: sticky;
            top: 0;
            z-index: 5;
        }}

        .hour-label {{
            background: var(--sidebar-bg);
            padding: 0.5rem;
            font-size: 0.65rem;
            color: var(--text-muted);
            text-align: right;
            border-right: 1px solid var(--border);
        }}

        .hour-slot {{
            background: var(--sidebar-bg);
            min-height: 80px;
            position: relative;
            padding: 2px;
        }}

        .hour-slot-today {{
            background: rgba(59, 130, 246, 0.03);
        }}

        .task-entry {{
            background: var(--primary);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 2px;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .task-entry:hover {{
            transform: scale(1.02);
            z-index: 10;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        /* Expandable Rows */
        .expandable-row {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            border-left: 3px solid transparent;
        }}

        .expandable-row:hover {{
            background: rgba(255, 255, 255, 0.03);
            border-left-color: var(--primary);
        }}

        .expandable-row.expanded {{
            background: rgba(255, 255, 255, 0.02);
            border-left-color: var(--primary);
        }}

        /* Toggle Switch */
        .switch {{
            position: relative;
            display: inline-block;
            width: 32px;
            height: 18px;
        }}

        .switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}

        .slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: var(--text-muted);
            opacity: 0.3;
            transition: .3s;
            border-radius: 18px;
        }}

        .slider:before {{
            position: absolute;
            content: "";
            height: 14px;
            width: 14px;
            left: 2px;
            bottom: 2px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }}

        input:checked + .slider {{
            background-color: var(--primary);
            opacity: 1;
        }}

        input:checked + .slider:before {{
            transform: translateX(14px);
        }}

        .skill-card {{
            position: relative;
            padding-bottom: 2.5rem !important;
            display: flex;
            flex-direction: column;
            height: 100%;
        }}

        .skill-id-badge {{
            position: absolute;
            bottom: 0.75rem;
            left: 1.25rem;
            font-size: 0.7rem;
            opacity: 0.6;
            font-family: 'JetBrains Mono', monospace;
        }}

        .skill-version-badge {{
            background: rgba(255, 255, 255, 0.05);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 500;
        }}

        .markdown-preview {{
            padding: 2rem;
            margin: 1.5rem 0;
            background: var(--sidebar-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.6;
            animation: fadeIn 0.3s ease;
            width: 100%;
            overflow-x: auto;
        }}

        .markdown-preview table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            border: 1px solid var(--border);
        }}

        .markdown-preview th, .markdown-preview td {{
            border: 1px solid var(--border);
            padding: 0.75rem;
            text-align: left;
        }}

        .markdown-preview th {{
            background: rgba(255, 255, 255, 0.05);
            font-weight: 600;
        }}

        .markdown-preview tr:nth-child(even) {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .preview-toggle-icon {{
            transition: transform 0.3s ease;
        }}

        .expanded .preview-toggle-icon {{
            transform: rotate(180deg);
        }}

        /* Maximized Modal */
        #editor-modal .modal-inner {{
            width: 95vw;
            height: 90vh;
            max-width: 1400px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            padding: 2rem;
        }}

        #editor-area {{
            flex-grow: 1;
            width: 100%;
            background: var(--chat-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            padding: 1.5rem;
            resize: none;
            outline: none;
            line-height: 1.5;
        }}

        @media screen and (max-width: 768px) {{
            #editor-modal .modal-inner {{
                width: 100vw;
                height: 100vh;
                max-width: none;
                border-radius: 0;
                padding: 1rem;
            }}

            .timeline-grid {{
                grid-template-columns: 50px repeat(7, 180px);
            }}
        }}
    </style>
</head>
<body>
    <div class="sidebar-overlay" id="sidebar-overlay" onclick="closeSidebar()"></div>
    <aside id="sidebar">
        <div class="agent-brand">
            <div class="agent-emoji-small">{agent_emoji}</div>
            <div class="agent-brand-text">
                <h2>{agent_name}</h2>
                <p title="{agent_description}">{agent_description}</p>
            </div>
        </div>

        <nav>
            <div class="nav-dropdown">
                <button class="nav-item active-primary" data-tab="chat" onclick="showTab('chat')">
                    <span>💬</span> Chat
                </button>
                <div class="nav-dropdown-content">
                    <button class="nav-item focus-ring" onclick="newChat()" style="border-bottom: 1px solid var(--border); border-radius: 8px 8px 0 0;">
                        <span>+</span> New Chat
                    </button>
                    <div id="chat-history-dropdown" style="padding-top: 0.5rem;">
                        <!-- Chat history items will be moved here -->
                    </div>
                </div>
            </div>
             <button class="nav-item" data-tab="knowledge" onclick="showTab('knowledge')"><span>📚</span> Knowledge</button>
            <button class="nav-item" data-tab="skills" onclick="showTab('skills')"><span>⚡</span> Skills</button>
            <button class="nav-item" data-tab="schedule" onclick="showTab('schedule')"><span>📅</span> Scheduled Tasks</button>
            <button class="nav-item" data-tab="files" onclick="showTab('files')"><span>📂</span> Files</button>
            <button class="nav-item" data-tab="config" onclick="showTab('config')"><span>⚙️</span> Configuration</button>
        </nav>

        <div style="margin-top: auto; padding-top: 1rem; border-top: 1px solid var(--border);">
            <div style="font-size: 0.65rem; color: var(--text-muted); text-align: center; opacity: 0.4; text-transform: uppercase; letter-spacing: 0.05em;">
                {agent_name} Dashboard v{agent_version}
            </div>
        </div>
    </aside>

    <main>
        <div class="mobile-header">
            <button class="hamburger" onclick="toggleSidebar()" aria-label="Toggle Menu">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
            </button>
            <div style="font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>{agent_emoji}</span> {agent_name}
            </div>
            <div style="width: 40px;"></div> <!-- Spacer for balance -->
        </div>
        <section id="chat-tab" class="tab-content" style="display: flex; flex-direction: column;">
            <div id="chat-history-container" style="padding: 1rem 2rem; border-bottom: 1px solid var(--border); background: rgba(255,255,255,0.02); display: none;">
                <h4 style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">Previous Chats</h4>
                <div id="chat-history-list" style="display: flex; gap: 0.5rem; overflow-x: auto; padding-bottom: 0.5rem;">
                    <!-- Chat history items will be loaded here -->
                </div>
            </div>
            
            <div id="chat-messages" style="flex-grow: 1;">
                <!-- Initial Welcome will be cleared when loading history -->
                <div class="message message-assistant" id="welcome-message">
                    <div class="message-header">
                        <div class="message-avatar">{agent_emoji}</div>
                    </div>
                    <div class="message-body">
                        <div class="message-bubble">
                            <div class="message-content">
                                <p>Greetings. I am <strong>{agent_name}</strong>. How may I assist you today?</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="chat-input-container">
                <div class="input-wrapper">
                    <textarea id="user-input" placeholder="Type a message..." rows="1" oninput="autoResize(this)"></textarea>
                    <button id="send-btn" class="send-btn" onclick="sendMessage()" title="Send">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </div>
            </div>
        </section>

        <!-- Management Sections -->
        <section id="files-tab" class="tab-content">
            <div class="content-inner">
                <div class="section-header">
                    <h2>Workspace Assets</h2>
                    <p>Live memory and generated artifacts.</p>
                </div>
                <div class="card">
                    <h3>📥 Generated Files</h3>
                    <ul id="generated-files"></ul>
                </div>
            </div>
        </section>

        <section id="config-tab" class="tab-content">
            <div class="content-inner">
                <div class="section-header">
                    <div>
                        <h2>Configuration</h2>
                        <p>Edit core agent parameters and identity.</p>
                    </div>
                    <div style="display: flex; gap: 0.75rem;">
                        <button class="btn btn-outline" onclick="reloadAgent()" style="padding: 0.5rem 1rem; border-color: var(--primary); color: var(--primary);">
                            <span>🔄</span> Sync Workspace
                        </button>
                        <button class="btn btn-outline" onclick="toggleTheme()" style="padding: 0.5rem 1rem;">
                            <span id="theme-icon">🌙</span> <span id="theme-text">Dark Mode</span>
                        </button>
                    </div>
                </div>
                <div class="card">
                    <h3>📄 System Files</h3>
                    <div id="config-files" style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem;"></div>
                </div>
            </div>
        </section>

        <section id="knowledge-tab" class="tab-content">
            <div class="content-inner">
                <div class="section-header">
                    <h2>Knowledge Bases</h2>
                    <p>Documentation and context modules (Skill Graphs) suffix <b>-docs</b>.</p>
                </div>
                <div id="knowledge-container"></div>
            </div>
        </section>

        <section id="skills-tab" class="tab-content">
            <div class="content-inner">
                <div class="section-header">
                    <h2>Skills</h2>
                    <p>Loaded tools and skill modules available to the agent.</p>
                </div>
                <div id="skills-container"></div>
            </div>
        </section>

        <section id="schedule-tab" class="tab-content">
            <div class="content-inner">
                <div class="section-header" style="display: flex; justify-content: space-between; align-items: flex-end;">
                    <div>
                        <h2>Scheduled Tasks</h2>
                        <p>Scheduled automation and background tasks.</p>
                    </div>
                    <div class="view-controls" style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <button class="btn btn-outline" id="btn-view-week" onclick="changeCalendarView('week')" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">Week</button>
                        <button class="btn btn-outline" id="btn-view-month" onclick="changeCalendarView('month')" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">Month</button>
                        <div style="margin-left: 1rem; display: flex; gap: 0.25rem;">
                            <button class="btn btn-outline" onclick="navigateCalendar(-1)" style="padding: 0.4rem 0.6rem;">←</button>
                            <button class="btn btn-outline" onclick="navigateCalendar(1)" style="padding: 0.4rem 0.6rem;">→</button>
                        </div>
                    </div>
                </div>

                <div class="card" style="margin-bottom: 2rem;">
                    <h3>📅 Schedule Overview</h3>
                    <div id="cron-calendar-view" style="margin-top: 1.5rem;">
                        <!-- Timeline grid will be generated here -->
                    </div>
                </div>

                <div class="card">
                    <h3>🔍 Task Details</h3>
                    <div id="cron-task-list"></div>
                </div>

                <div class="card">
                    <h3>📜 Execution History (CRON_LOG.md)</h3>
                    <div id="cron-log-content" style="max-height: 400px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
                        Loading history...
                    </div>
                </div>
            </div>
        </section>
    </main>

    <div id="editor-modal">
        <div class="modal-inner">
            <h2 id="editor-title">File Editor</h2>
            <textarea id="editor-area" spellcheck="false"></textarea>
            <div style="display: flex; justify-content: flex-end; gap: 1rem;">
                <button class="btn btn-outline" onclick="closeEditor()">Cancel</button>
                <button class="btn" onclick="saveFile()">Save Changes</button>
            </div>
        </div>
    </div>

    <script>
        // State
        const AGENT_NAME = "{agent_name}";
        const AGENT_EMOJI = "{agent_emoji}";
        let messages = [];
        let currentChatId = null;

        // UI Controls
        function autoResize(textarea) {{
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        }}

        function showTab(tabId) {{
            console.log('Showing tab:', tabId);
            document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('nav .nav-item').forEach(el => el.classList.remove('active-primary'));

            const tab = document.getElementById(`${{tabId}}-tab`);
            if (!tab) {{
                console.error('Tab not found:', `${{tabId}}-tab`);
                return;
            }}
            tab.style.display = (tabId === 'chat') ? 'flex' : 'block';

            // Highlight the correct nav item by data-tab
            const btn = document.querySelector(`nav .nav-item[data-tab="${{tabId}}"]`);
            if (btn) btn.classList.add('active-primary');
            else console.warn('Nav button not found for tab:', tabId);

            if (tabId === 'files' || tabId === 'config') loadFiles();
            if (tabId === 'skills' || tabId === 'knowledge') loadSkills();
            if (tabId === 'schedule') loadCron();

            // Close sidebar on mobile after navigation
            if (window.innerWidth <= 768) closeSidebar();
        }}

        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebar-overlay');
            sidebar.classList.toggle('sidebar-open');
            overlay.classList.toggle('active');
        }}

        function closeSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebar-overlay');
            sidebar.classList.remove('sidebar-open');
            overlay.classList.remove('active');
        }}

        // Theme Management
        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const target = current === 'dark' ? 'light' : 'dark';
            setTheme(target);
        }}

        function setTheme(theme) {{
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('ag-ui-theme', theme);

            const icon = document.getElementById('theme-icon');
            const text = document.getElementById('theme-text');
            if (theme === 'dark') {{
                icon.innerText = '🌙';
                text.innerText = 'Dark Mode';
            }} else {{
                icon.innerText = '☀️';
                text.innerText = 'Light Mode';
            }}
        }}

        function initTheme() {{
            const saved = localStorage.getItem('ag-ui-theme') || 'dark';
            setTheme(saved);
        }}

        // Native Chat Implementation
        async function sendMessage() {{
            const input = document.getElementById('user-input');
            const btn = document.getElementById('send-btn');
            const text = input.value.trim();

            if (!text || btn.disabled) return;

            // Clear welcome message if it's the first message
            const welcome = document.getElementById('welcome-message');
            if (welcome) welcome.remove();

            // Add user message to UI
            addMessage('user', text);
            input.value = '';
            input.style.height = 'auto';
            btn.disabled = true;

            try {{
                // Save immediately to establish session if needed
                if (!currentChatId) await persistChat();

                const payload = {{
                    trigger: 'submit-message',
                    id: `req-${{Date.now()}}`,
                    messages: messages.map(m => ({{
                        id: m.id.toString(),
                        role: m.role,
                        parts: m.parts && m.parts.length ? m.parts : [{{ type: 'text', text: m.content }}]
                    }}))
                }};

                const response = await fetch('/api/chat', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                        'x-vercel-ai-ui-message-stream': 'v1'
                    }},
                    body: JSON.stringify(payload)
                }});

                if (!response.ok) throw new Error('API Error: ' + response.status);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMsg = addMessage('assistant', '');

                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, {{ stream: true }});
                    const lines = chunk.split('\\n');

                    for (const line of lines) {{
                        if (!line.startsWith('data: ')) continue;
                        try {{
                            const jsonData = JSON.parse(line.substring(6));
                            handleStreamChunk(jsonData, assistantMsg);
                        }} catch (e) {{}}
                    }}
                }}

                // Persist after assistant finished
                await persistChat();
                loadHistory(); // Refresh history list

            }} catch (e) {{
                console.error(e);
                addMessage('assistant', '⚠️ Connection error: ' + e.message);
            }} finally {{
                btn.disabled = false;
            }}
        }}

        function addMessage(role, content) {{
            const container = document.getElementById('chat-messages');
            const msgObj = {{ role: role, content: content, parts: [], id: Date.now() }};
            messages.push(msgObj);

            const div = document.createElement('div');
            div.className = `message message-${{role}}`;
            div.innerHTML = `
                <div class="message-header">
                    <div class="message-avatar">${{role === 'user' ? '👤' : AGENT_EMOJI}}</div>
                </div>
                <div class="message-body">
                    <div class="message-bubble">
                        <div class="message-content" id="msg-content-${{msgObj.id}}">
                            ${{marked.parse(content)}}
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return msgObj;
        }}

        function handleStreamChunk(data, msgObj) {{
            const contentEl = document.getElementById(`msg-content-${{msgObj.id}}`);

            if (data.type === 'text-delta') {{
                msgObj.content += data.delta;
                contentEl.innerHTML = marked.parse(msgObj.content);
            }} else if (data.type === 'reasoning-delta') {{
                let reasonEl = contentEl.querySelector('.reasoning');
                if (!reasonEl) {{
                    reasonEl = document.createElement('div');
                    reasonEl.className = 'reasoning';
                    contentEl.prepend(reasonEl);
                }}
                reasonEl.innerText += data.delta;
            }} else if (data.type === 'tool-input-start') {{
                const toolId = data.toolCallId;
                const toolDiv = document.createElement('div');
                toolDiv.className = 'tool-call';
                toolDiv.id = `tool-${{toolId}}`;
                toolDiv.innerHTML = `
                    <div class="tool-header" onclick="toggleTool('${{toolId}}')">
                        <div class="tool-name"><span>🛠️</span> Executing <strong>${{data.toolName}}</strong></div>
                        <div class="tool-status tool-status-running">Running</div>
                    </div>
                    <div class="tool-body" id="tool-body-${{toolId}}" style="display: none;">
                        <pre id="tool-input-${{toolId}}"></pre>
                        <pre id="tool-output-${{toolId}}" style="color: var(--accent); margin-top: 1rem; border-top: 1px solid var(--border); padding-top: 1rem; display: none;"></pre>
                    </div>
                `;
                contentEl.appendChild(toolDiv);
            }} else if (data.type === 'tool-input-delta') {{
                const inputEl = document.getElementById(`tool-input-${{data.toolCallId}}`);
                if (inputEl) inputEl.innerText += data.inputTextDelta;
            }} else if (data.type === 'tool-output-available') {{
                const statusEl = document.querySelector(`#tool-${{data.toolCallId}} .tool-status`);
                if (statusEl) {{
                    statusEl.innerText = 'Success';
                    statusEl.className = 'tool-status tool-status-done';
                }}
                const outEl = document.getElementById(`tool-output-${{data.toolCallId}}`);
                if (outEl) {{
                    outEl.innerText = (typeof data.output === 'string') ? data.output : JSON.stringify(data.output, null, 2);
                    outEl.style.display = 'block';
                }}
            }}

            const container = document.getElementById('chat-messages');
            container.scrollTop = container.scrollHeight;
        }}

        function toggleTool(id) {{
            const body = document.getElementById(`tool-body-${{id}}`);
            body.style.display = body.style.display === 'none' ? 'block' : 'none';
        }}

        // Persistence & History
        async function persistChat() {{
            const res = await fetch('/api/enhanced/chats', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    id: currentChatId,
                    messages: messages
                }})
            }});
            const data = await res.json();
            if (data.status === 'success') {{
                currentChatId = data.id;
            }}
        async function loadHistory() {{
            const res = await fetch('/api/enhanced/chats');
            const data = await res.json();
            
            // 1. Sidebar Dropdown (Legacy support)
            const dropdown = document.getElementById('chat-history-dropdown');
            if (dropdown) {{
                if (!data || data.length === 0) {{
                    dropdown.innerHTML = '<div style="font-size: 0.75rem; color: var(--text-muted); padding: 0.5rem 1rem;">No history yet</div>';
                }} else {{
                    dropdown.innerHTML = data.map(chat => `
                        <div class="history-item ${{currentChatId === chat.id ? 'active' : ''}}"
                             onclick="loadChat('${{chat.id}}')"
                             style="padding-left: 1rem; font-size: 0.78rem;">
                            <span style="flex-grow: 1; overflow: hidden; text-overflow: ellipsis;">${{chat.title}}</span>
                        </div>
                    `).join('');
                }}
            }}

            // 2. Chat Tab Integrated List
            const historyList = document.getElementById('chat-history-list');
            const historyContainer = document.getElementById('chat-history-container');
            if (historyList) {{
                if (!data || data.length === 0) {{
                    historyContainer.style.display = 'none';
                }} else {{
                    historyContainer.style.display = 'block';
                    historyList.innerHTML = data.map(chat => `
                        <div class="card history-card ${{currentChatId === chat.id ? 'active' : ''}}" 
                             onclick="loadChat('${{chat.id}}')"
                             style="padding: 0.75rem 1rem; min-width: 180px; cursor: pointer; border: 1px solid var(--border); background: var(--card-bg); border-radius: 8px; flex-shrink: 0; position: relative;">
                            <div style="font-size: 0.8rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${{chat.title}}</div>
                            <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 0.25rem;">${{new Date(chat.updated_at).toLocaleDateString()}}</div>
                            <button class="rename-btn" style="position: absolute; top: 5px; right: 5px; font-size: 0.6rem; opacity:0; transition: opacity 0.2s;" 
                                onclick="event.stopPropagation(); renameChat('${{chat.id}}', '${{chat.title.replace(/'/g, "\\'")}}')">✏️</button>
                        </div>
                    `).join('');
                }}
            }}
        }}

        async function renameChat(id, oldTitle) {{
            const newTitle = prompt('Enter new chat title:', oldTitle);
            if (!newTitle || newTitle === oldTitle) return;

            try {{
                const res = await fetch(`/api/enhanced/chats/${{id}}/title`, {{
                    method: 'PUT',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ title: newTitle }})
                }});
                if (res.ok) loadHistory();
            }} catch (e) {{
                console.error('Failed to rename chat:', e);
            }}
        }}

        async function loadChat(id) {{
            const res = await fetch(`/api/enhanced/chats/${{id}}`);
            const data = await res.json();

            currentChatId = id;
            messages = data.messages || [];

            const container = document.getElementById('chat-messages');
            container.innerHTML = '';

            messages.forEach(m => {{
                const div = document.createElement('div');
                div.className = `message message-${{m.role}}`;
                div.innerHTML = `
                    <div class="message-header">
                        <div class="message-avatar">${{m.role === 'user' ? '👤' : AGENT_EMOJI}}</div>
                    </div>
                    <div class="message-body">
                        <div class="message-bubble">
                            <div class="message-content" id="msg-content-${{m.id}}">
                                ${{marked.parse(m.content)}}
                            </div>
                        </div>
                    </div>
                `;
                container.appendChild(div);
            }});

            container.scrollTop = container.scrollHeight;
            loadHistory();
            showTab('chat');
        }}

        function newChat() {{
            currentChatId = null;
            messages = [];
            document.getElementById('chat-messages').innerHTML = `
                <div class="message message-assistant" id="welcome-message">
                    <div class="message-header">
                        <div class="message-avatar">${{AGENT_EMOJI}}</div>
                    </div>
                    <div class="message-body">
                        <div class="message-bubble">
                            <div class="message-content">
                                <p>Greetings. I am <strong>${{AGENT_NAME}}</strong>. Starting a fresh session. How can I help?</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            loadHistory();
            showTab('chat');
        }}

        async function fetchAPI(endpoint, options = {{}}) {{
            const res = await fetch(`/api/enhanced/${{endpoint}}`, options);
            return res.json();
        }}

        // Scheduling State and Functions
        let currentScheduleDate = new Date();
        let calendarView = 'week'; // 'week' or 'month'

        function navigateCalendar(offset) {{
            if (calendarView === 'week') {{
                currentScheduleDate.setDate(currentScheduleDate.getDate() + (offset * 7));
            }} else {{
                currentScheduleDate.setMonth(currentScheduleDate.getMonth() + offset);
            }}
            loadCron();
        }}

        function changeCalendarView(view) {{
            calendarView = view;
            document.querySelectorAll('.view-controls .btn').forEach(b => b.classList.remove('active-primary'));
            const activeBtn = document.getElementById(`btn-view-${{view}}`);
            if (activeBtn) activeBtn.classList.add('active-primary');
            loadCron();
        }}

        async function loadSkills() {{
            try {{
                const data = await fetchAPI('skills');
                const skillsContainer = document.getElementById('skills-container');
                const knowledgeContainer = document.getElementById('knowledge-container');
                
                if (!skillsContainer || !knowledgeContainer) return;

                const skills = data.filter(s => !s.id.endsWith('-docs'));
                const knowledge = data.filter(s => s.id.endsWith('-docs'));

                const renderSkill = (s) => `
                    <div class="card skill-card" style="padding: 1.25rem; background: var(--card-bg);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                <div style="font-size: 1.5rem;">\${s.id.endsWith('-docs') ? '📚' : '⚡'}</div>
                                <div>
                                    <h4 style="margin: 0; font-size: 0.95rem;">\${s.name}</h4>
                                    <span class="skill-version-badge">v\${s.version || '0.1.0'}</span>
                                </div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" \${s.enabled ? 'checked' : ''} onclick="toggleSkill('\${s.id}')">
                                <span class="slider"></span>
                            </label>
                        </div>
                        <p style="font-size: 0.82rem; color: var(--text-muted); margin: 0; flex-grow: 1;">\${s.description}</p>
                        <div class="skill-id-badge">ID: \${s.id}</div>
                    </div>
                `;

                skillsContainer.innerHTML = skills.length > 0 
                    ? `<div class="grid">\${skills.map(renderSkill).join('')}</div>`
                    : '<p style="text-align: center; color: var(--text-muted); padding: 2rem;">No capability skills loaded.</p>';

                knowledgeContainer.innerHTML = knowledge.length > 0
                    ? `<div class="grid">\${knowledge.map(renderSkill).join('')}</div>`
                    : '<p style="text-align: center; color: var(--text-muted); padding: 2rem;">No knowledge bases (skill graphs) found.</p>';
                
            }} catch (e) {{
                console.error('Failed to load skills/knowledge:', e);
            }}
        }}

        async function toggleSkill(skillId) {{
            try {{
                const res = await fetch(`/api/enhanced/skills/${{skillId}}/toggle`, {{ method: 'POST' }});
                const data = await res.json();
                if (data.status === 'success') {{
                    console.log(`Skill ${{skillId}} toggled: ${{data.enabled}}`);
                }}
            }} catch (e) {{
                console.error('Toggle failed:', e);
            }}
        }}

        async function reloadAgent() {
            if (!confirm('Are you sure you want to re-sync the workspace and reload the agent?')) return;
            try {{
                const res = await fetch('/api/enhanced/reload', {{ method: 'POST' }});
                const data = await res.json();
                if (data.status === 'success') {{
                    alert('Agent reloaded successfully!');
                    window.location.reload();
                }} else {{
                    alert('Reload failed: ' + data.detail);
                }}
            }} catch (e) {{
                console.error('Reload failed:', e);
                alert('Reload failed. Check console for details.');
            }}
        }}

        async function loadFiles() {{
            try {{
                const data = await fetchAPI('files');
                console.log('Loaded files:', data);

                const configUl = document.getElementById('config-files');
                const generatedUl = document.getElementById('generated-files');

                if (data.config && configUl) {{
                    configUl.innerHTML = data.config.map(f => `
                        <div class="list-item-wrapper" style="margin-bottom: 0.5rem; width: 100%;">
                            <div class="expandable-row list-item" id="row-${{f}}" onclick="toggleMarkdownPreview('${{f}}')" style="margin-bottom: 0;">
                                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                                        <span class="preview-toggle-icon">▶</span>
                                        <span style="font-weight: 500;">${{f}}</span>
                                    </div>
                                    <button class="btn btn-outline" style="padding: 0.3rem 0.6rem;" onclick="event.stopPropagation(); editFile('${{f}}')">Edit</button>
                                </div>
                            </div>
                            <div id="preview-${{f}}" class="markdown-preview" style="display: none; width: 100%;" onclick="event.stopPropagation()">
                                <div class="loading-spinner" style="font-size: 0.8rem; opacity: 0.5;">Rendering ${{f}}...</div>
                            </div>
                        </div>
                    `).join('');
                }}

                if (data.generated && generatedUl) {{
                    generatedUl.innerHTML = data.generated.map(f => `
                        <li class="list-item">
                            <span>${{f}}</span>
                            <a href="/api/enhanced/download/${{f}}" class="btn btn-outline" style="padding: 0.3rem 0.6rem;">Get</a>
                        </li>
                    `).join('') || '<p style="color: var(--text-muted); opacity: 0.6;">None.</p>';
                }}
            }} catch (e) {{
                console.error('Failed to load files:', e);
            }}
        }}

        async function toggleMarkdownPreview(filename) {{
            const row = document.getElementById(`row-${{filename}}`);
            const preview = document.getElementById(`preview-${{filename}}`);

            if (preview.style.display === 'none') {{
                row.classList.add('expanded');
                preview.style.display = 'block';

                const data = await fetchAPI(`files/${{filename}}`);
                preview.innerHTML = marked.parse(data.content);
            }} else {{
                row.classList.remove('expanded');
                preview.style.display = 'none';
            }}
        }}

        let currentEditingFile = '';
        async function editFile(filename) {{
            currentEditingFile = filename;
            const data = await fetchAPI(`files/${{filename}}`);
            document.getElementById('editor-title').innerText = filename;
            document.getElementById('editor-area').value = data.content;
            document.getElementById('editor-modal').style.display = 'flex';
        }}

        function closeEditor() {{ document.getElementById('editor-modal').style.display = 'none'; }}
        async function saveFile() {{
            const content = document.getElementById('editor-area').value;
            await fetchAPI(`files/${{currentEditingFile}}`, {{
                method: 'PUT',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ content }})
            }});
            closeEditor();
            loadFiles();
        }}

        async function loadCron() {{
            const data = await fetchAPI('cron/calendar');
            const logData = await fetchAPI('files/CRON_LOG.md');
            
            const logEl = document.getElementById('cron-log-content');
            if (logEl) logEl.innerText = logData.content || 'No log history found.';

            const calView = document.getElementById('cron-calendar-view');
            if (calView) {{
                calView.innerHTML = '';
                
                if (calendarView === 'week') {{ calView.removeAttribute('style');
                    calView.className = 'timeline-grid';
                    const days = ['Time', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
                    
                    // Find start of week for currentScheduleDate
                    const startOfWeek = new Date(currentScheduleDate);
                    const day = startOfWeek.getDay();
                    const diff = startOfWeek.getDate() - day + (day === 0 ? -6 : 1);
                    startOfWeek.setDate(diff);
                    
                    days.forEach((dayName, idx) => {{
                        const header = document.createElement('div');
                        header.className = 'timeline-header';
                        if (idx > 0) {{
                            const d = new Date(startOfWeek);
                            d.setDate(startOfWeek.getDate() + idx - 1);
                            header.innerHTML = `${{dayName}}<br><span style="font-size: 0.6rem; font-weight: normal; opacity: 0.6;">${{d.getMonth()+1}}/${{d.getDate()}}</span>`;
                            const today = new Date();
                            if (d.toDateString() === today.toDateString()) header.style.color = 'var(--primary)';
                        }} else {{ calView.removeAttribute('style');
                            header.innerText = dayName;
                        }}
                        calView.appendChild(header);
                    }});

                    for (let hour = 0; hour < 24; hour++) {{
                        const label = document.createElement('div');
                        label.className = 'hour-label';
                        label.innerText = hour === 0 ? '12 AM' : (hour < 12 ? `${{hour}} AM` : (hour === 12 ? '12 PM' : `${{hour-12}} PM`));
                        calView.appendChild(label);

                        for (let dayIdx = 0; dayIdx < 7; dayIdx++) {{
                            const slot = document.createElement('div');
                            slot.className = 'hour-slot';
                            
                            const slotDate = new Date(startOfWeek);
                            slotDate.setDate(startOfWeek.getDate() + dayIdx);
                            if (slotDate.toDateString() === new Date().toDateString()) slot.classList.add('hour-slot-today');
                            
                            const activeTasks = data.filter(t => t.active);
                            activeTasks.forEach((t) => {{
                                const intervalHours = Math.max(1, Math.floor(t.interval_min / 60));
                                // Show in slot if hour divides by interval, and we want to show it daily
                                if (hour % intervalHours === 0) {{
                                    const entry = document.createElement('div');
                                    entry.className = 'task-entry';
                                    entry.title = `${{t.name}}\nEvery ${{t.interval_min}}m\nNext: ${{t.next_approx}}`;
                                    entry.innerHTML = `<span>⚙️</span> <span style="overflow:hidden; text-overflow:ellipsis;">${{t.name}}</span>`;
                                    entry.onclick = () => alert(`Task: ${{t.name}}\nInterval: ${{t.interval_min}}m\nPrompt: ${{t.prompt}}`);
                                    slot.appendChild(entry);
                                }}
                            }});
                            calView.appendChild(slot);
                        }}
                    }}
                }} else {{
                    calView.className = '';
                    calView.style.display = 'grid';
                    calView.style.gridTemplateColumns = 'repeat(7, 1fr)';
                    calView.style.gap = '1px';
                    calView.style.background = 'var(--border)';
                    calView.style.borderRadius = '12px';
                    calView.style.overflow = 'hidden';
                    calView.style.border = '1px solid var(--border)';
                    
                    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
                    const monthTitle = document.createElement('div');
                    monthTitle.style.gridColumn = 'span 7';
                    monthTitle.style.padding = '1rem';
                    monthTitle.style.textAlign = 'center';
                    monthTitle.style.fontWeight = '700';
                    monthTitle.style.background = 'var(--sidebar-bg)';
                    monthTitle.style.color = 'var(--primary)';
                    monthTitle.innerText = `${{monthNames[currentScheduleDate.getMonth()]}} ${{currentScheduleDate.getFullYear()}}`;
                    calView.appendChild(monthTitle);

                    ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].forEach(day => {{
                        const h = document.createElement('div');
                        h.style.padding = '0.75rem';
                        h.style.textAlign = 'center';
                        h.style.background = 'var(--sidebar-bg)';
                        h.style.fontSize = '0.7rem';
                        h.style.color = 'var(--text-muted)';
                        h.style.fontWeight = '600';
                        h.style.borderBottom = '1px solid var(--border)';
                        h.innerText = day;
                        calView.appendChild(h);
                    }});

                    const firstDay = new Date(currentScheduleDate.getFullYear(), currentScheduleDate.getMonth(), 1).getDay();
                    const daysInMonth = new Date(currentScheduleDate.getFullYear(), currentScheduleDate.getMonth() + 1, 0).getDate();

                    for (let i = 0; i < firstDay; i++) {{
                        const empty = document.createElement('div');
                        empty.style.background = 'var(--sidebar-bg)';
                        empty.style.minHeight = '100px';
                        empty.style.opacity = '0.3';
                        calView.appendChild(empty);
                    }}

                    for (let d = 1; d <= daysInMonth; d++) {{
                        const dayCell = document.createElement('div');
                        dayCell.style.background = 'var(--sidebar-bg)';
                        dayCell.style.minHeight = '120px';
                        dayCell.style.padding = '0.5rem';
                        dayCell.style.position = 'relative';
                        dayCell.style.display = 'flex';
                        dayCell.style.flexDirection = 'column';
                        dayCell.style.gap = '0.4rem';
                        
                        const isToday = new Date().toDateString() === new Date(currentScheduleDate.getFullYear(), currentScheduleDate.getMonth(), d).toDateString();
                        
                        dayCell.innerHTML = `<div style="font-size: 0.75rem; font-weight: 700; color: ${{isToday ? 'var(--primary)' : 'var(--text-muted)'}}; margin-bottom: 0.2rem;">${{d}}</div>`;
                        
                        const activeTasks = data.filter(t => t.active);
                        if (activeTasks.length > 0) {{
                            // Deduplicate by name for month view summary
                            const taskNames = [...new Set(activeTasks.map(t => t.name))];
                            taskNames.slice(0, 3).forEach(name => {{
                                const entry = document.createElement('div');
                                entry.className = 'task-entry';
                                entry.style.padding = '2px 6px';
                                entry.style.fontSize = '0.65rem';
                                entry.innerHTML = `<span>⚙️</span> <span style="overflow:hidden; text-overflow:ellipsis;">${{name}}</span>`;
                                dayCell.appendChild(entry);
                            }});
                            if (taskNames.length > 3) {{
                                const more = document.createElement('div');
                                more.style.fontSize = '0.6rem';
                                more.style.color = 'var(--text-muted)';
                                more.style.textAlign = 'center';
                                more.innerText = `+${{taskNames.length - 3}} more`;
                                dayCell.appendChild(more);
                            }}
                        }}
                        calView.appendChild(dayCell);
                    }}
                }}
            }}

            const listContainer = document.getElementById('cron-task-list');
            if (listContainer) {{
                if (data.length === 0) {{ 
                    listContainer.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 2rem;">No active tasks found in CRON.md.</p>'; 
                    return; 
                }}
                listContainer.innerHTML = data.map(t => `
                    <div class="list-item" style="border-left: 3px solid ${{t.active ? 'var(--accent)' : 'var(--error)'}};">
                        <div style="flex-grow: 1;">
                            <div style="font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                                ${{t.name}} 
                                <span class="badge" style="background: rgba(255,255,255,0.05); color: var(--text-muted);">${{t.id}}</span>
                                ${{t.active ? '<span style="color: var(--accent); font-size: 0.7rem;">● Active</span>' : '<span style="color: var(--error); font-size: 0.7rem;">○ Inactive</span>'}}
                            </div>
                            <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 400px;">
                                Prompt: ${{t.prompt}}
                            </div>
                        </div>
                        <div style="text-align: right; min-width: 150px;">
                            <div style="font-size: 0.75rem; color: var(--primary); font-weight: 600;">Next approx</div>
                            <div style="font-size: 0.85rem; font-weight: 500;">${{t.next_approx}}</div>
                        </div>
                    </div>
                `).join('');
            }}
        }}

        window.onload = () => {{
            initTheme();
            loadHistory();
            changeCalendarView('week');
        }};

        document.getElementById('user-input').addEventListener('keydown', (e) => {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                sendMessage();
            }}
        }});
    </script>
</body>
</html>
"""
