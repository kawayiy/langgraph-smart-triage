# Import Gradio for building the interactive frontend
import gradio as gr
# Import `requests` for sending HTTP requests
import requests
# Import `json` for working with JSON payloads
import json
# Import `logging` for application logs
import logging
# Import `re` for regular expressions
import re
# Import `uuid` for generating unique identifiers
import uuid
# Import `datetime` for date and time handling
from datetime import datetime
# Import `bcrypt` for password hashing and verification
import bcrypt

# Configure basic logging with INFO level and a standard format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger for the current module
logger = logging.getLogger(__name__)

# Backend service URL
url = "http://localhost:8013/v1/chat/completions"
# HTTP headers declaring JSON content
headers = {"Content-Type": "application/json"}

# Whether to stream responses
stream_flag = True # False

# In-memory user database
users_db = {}
# Mapping from username to user ID
user_id_map = {}

# Generate a unique user ID
def generate_unique_user_id(username):
    # Generate a new UUID if the username has not been seen before
    if username not in user_id_map:
        user_id = str(uuid.uuid4())
        # Ensure the generated ID is not already in use
        while user_id in user_id_map.values():
            user_id = str(uuid.uuid4())
        # Store the username-to-ID mapping
        user_id_map[username] = user_id
    # Return the unique ID for this user
    return user_id_map[username]

# Generate a unique conversation ID
def generate_unique_conversation_id(username):
    # Build the conversation ID from the username and a UUID
    return f"{username}_{uuid.uuid4()}"

# Send a message, handle user input, and fetch the backend response
def send_message(user_message, history, user_id, conversation_id, username):
    # Build the request payload sent to the backend
    data = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": user_id,
        "conversationId": conversation_id
    }

    # Update chat history with the user message and a temporary placeholder reply
    history = history + [["user", user_message], ["assistant", "正在生成回复..."]]
    # First yield: return the current chat history and keep the title unchanged
    yield history, history, None

    # Use the first message to initialize the conversation title
    if username and conversation_id:
        if not users_db[username]["conversations"][conversation_id].get("title_set", False):
            new_title = user_message[:20] if len(user_message) > 20 else user_message
            users_db[username]["conversations"][conversation_id]["title"] = new_title
            users_db[username]["conversations"][conversation_id]["title_set"] = True

    # Helper to format the response text
    def format_response(full_text):
        # Replace `<think>` with a bold "thinking process" heading
        formatted_text = re.sub(r'<think>', '**思考过程**：\n', full_text)
        # Replace `</think>` with a bold "final answer" heading
        formatted_text = re.sub(r'</think>', '\n\n**最终回复**：\n', full_text)
        # Return the trimmed formatted text
        return formatted_text.strip()

    # Streaming output
    if stream_flag:
        assistant_response = ""
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str:
                            logger.info(f"收到空字符串，跳过...")
                            continue
                        # logger.info(f"Received json_str: {json_str}")
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    # Format the response chunk in real time
                                    formatted_content = format_response(content)
                                    logger.info(f"接收数据:{formatted_content}")
                                    assistant_response += formatted_content
                                    updated_history = history[:-1] + [["assistant", assistant_response]]
                                    yield updated_history, updated_history, None
                                if response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    logger.info(f"接收JSON数据结束")
                                    break
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {e}")
                                yield history[:-1] + [["assistant", "解析响应时出错，请稍后再试。"]]
                                break
                        else:
                            logger.info(f"无效JSON格式: {json_str}")
                    else:
                        logger.info(f"收到空行")
                else:
                    logger.info("流式响应结束但未明确结束")
                    yield history[:-1] + [["assistant", "未收到完整响应。"]]
        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            yield history[:-1] + [["assistant", "请求失败，请稍后再试。"]]

    # Non-streaming output
    else:
        # Send a POST request to the backend
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # Parse the response as JSON
        response_json = response.json()
        # Extract the assistant's reply
        assistant_content = response_json['choices'][0]['message']['content']
        # Format the assistant response
        formatted_content = format_response(assistant_content)
        # Replace the placeholder reply with the formatted assistant output
        updated_history = history[:-1] + [["assistant", formatted_content]]
        # Second yield: return the updated history while keeping the title unchanged
        yield updated_history, updated_history, None

# Register a user. Passwords are stored as bcrypt hashes, never plaintext
def register(username, password):
    # Reject duplicate usernames
    if username in users_db:
        return "用户名已存在！"
    # Generate a unique user ID
    user_id = generate_unique_user_id(username)
    # Hash the password with bcrypt before storing it
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    # Store the new user record with the password hash only
    users_db[username] = {"password_hash": password_hash, "user_id": user_id, "conversations": {}}
    # Return the success message
    return "注册成功！请关闭弹窗并登录。"

# Log a user in by validating the bcrypt password hash
def login(username, password):
    # Validate that the username exists and the password hash matches
    if username in users_db and bcrypt.checkpw(
        password.encode("utf-8"), users_db[username]["password_hash"]
    ):
        # Get the user ID
        user_id = users_db[username]["user_id"]
        # Generate a new conversation ID
        conversation_id = generate_unique_conversation_id(username)
        # Use the current time as the conversation creation time
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Create a new conversation record for the user
        users_db[username]["conversations"][conversation_id] = {
            "history": [],
            "title": "创建新的聊天",
            "create_time": create_time,
            "title_set": False
        }
        # Return the login result and related data
        return True, username, user_id, conversation_id, "登录成功！"
    # Return an error when login fails
    return False, None, None, None, "用户名或密码错误！"

# Create a new conversation
def new_conversation(username):
    # Ask the user to log in first
    if username not in users_db:
        return "请先登录！", None
    # Generate a new conversation ID
    conversation_id = generate_unique_conversation_id(username)
    # Use the current time as the conversation creation time
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Store the new conversation record
    users_db[username]["conversations"][conversation_id] = {
        "history": [],
        "title": "创建新的聊天",
        "create_time": create_time,
        "title_set": False
    }
    # Return the success message and the new conversation ID
    return "新会话创建成功！", conversation_id

# Get the conversation list
def get_conversation_list(username):
    # Return a default option if the user is not logged in or has no conversations
    if username not in users_db or not users_db[username]["conversations"]:
        return ["请选择历史会话"]
    # Initialize the conversation list
    conv_list = []
    # Iterate over all user conversations
    for conv_id, details in users_db[username]["conversations"].items():
        # Read the conversation title, defaulting to "Untitled Conversation"
        title = details.get("title", "未命名会话")
        # Read the conversation creation time
        create_time = details.get("create_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # Append the combined title and time to the list
        conv_list.append(f"{title} - {create_time}")
    # Return the list with the default option prepended
    return ["请选择历史会话"] + conv_list

# Extract the conversation ID from a selected option
def extract_conversation_id(selected_option, username):
    # Return `None` for the default option or unauthenticated users
    if selected_option == "请选择历史会话" or not username in users_db:
        return None
    # Iterate over all user conversations
    for conv_id, details in users_db[username]["conversations"].items():
        # Read the conversation title and creation time
        title = details.get("title", "未命名会话")
        create_time = details.get("create_time", "")
        # Return the matching conversation ID
        if f"{title} - {create_time}" == selected_option:
            return conv_id
    # Return `None` when no match is found
    return None

# Load conversation history
def load_conversation(username, selected_option):
    # Return empty history for the default option or unauthenticated users
    if selected_option == "请选择历史会话" or not username in users_db:
        return []
    # Extract the conversation ID from the selected option
    conversation_id = extract_conversation_id(selected_option, username)
    # Return the matching chat history if the conversation exists
    if conversation_id in users_db[username]["conversations"]:
        return users_db[username]["conversations"][conversation_id]["history"]
    # Otherwise return empty history
    return []

# Build the frontend UI with Gradio Blocks
with gr.Blocks(title="聊天助手", css="""
    .login-container { max-width: 400px; margin: 0 auto; padding-top: 100px; }
    .modal { position: fixed; top: 20%; left: 50%; transform: translateX(-50%); background: white; padding: 20px; max-width: 400px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-radius: 8px; z-index: 1000; }
    .chat-area { padding: 20px; height: 80vh; }
    .header { display: flex; justify-content: space-between; align-items: center; padding: 10px; }
    .header-btn { margin-left: 10px; padding: 5px 10px; font-size: 14px; }
""") as demo:
    # State used to track login status
    logged_in = gr.State(False)
    # State holding the current username
    current_user = gr.State(None)
    # State holding the current user ID
    current_user_id = gr.State(None)
    # State holding the current conversation ID
    current_conversation = gr.State(None)
    # State holding the chat history
    chatbot_history = gr.State([])
    # State holding the conversation title
    conversation_title = gr.State("创建新的聊天")

    # Login page layout, visible by default
    with gr.Column(visible=True, elem_classes="login-container") as login_page:
        # Display the page title
        gr.Markdown("## 聊天助手")
        # Username input
        login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        # Password input with hidden content
        login_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        # Row for login and register buttons
        with gr.Row():
            # Login button
            login_button = gr.Button("登录", variant="primary")
            # Register button
            register_button = gr.Button("注册", variant="secondary")
        # Non-editable login result output
        login_output = gr.Textbox(label="结果", interactive=False)

    # Chat page layout, hidden by default
    with gr.Column(visible=False) as chat_page:
        # Header layout with welcome text and actions
        with gr.Row(elem_classes="header"):
            # Welcome text, empty by default
            welcome_text = gr.Markdown("### 欢迎，")
            # Row for header buttons
            with gr.Row():
                # New conversation button
                new_conv_button = gr.Button("新建会话", elem_classes="header-btn", variant="secondary")
                # History button
                history_button = gr.Button("历史会话", elem_classes="header-btn", variant="secondary")
                # Logout button
                logout_button = gr.Button("退出登录", elem_classes="header-btn", variant="secondary")

        # Chat area layout
        with gr.Column(elem_classes="chat-area"):
            # Conversation title display
            title_display = gr.Markdown("## 会话标题", elem_id="title-display")
            # Chat panel
            chatbot = gr.Chatbot(label="聊天对话", height=450)
            # Row for message input and send button
            with gr.Row():
                # Message input
                message = gr.Textbox(label="消息", placeholder="输入消息并按 Enter 发送", scale=8, container=False)
                # Send button
                send = gr.Button("发送", scale=2)

    # Registration modal, hidden by default
    with gr.Column(visible=False, elem_classes="modal") as register_modal:
        # Registration username input
        reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        # Registration password input
        reg_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        # Row for submit and close buttons
        with gr.Row():
            # Submit registration button
            reg_button = gr.Button("提交注册", variant="primary")
            # Close dialog button
            close_button = gr.Button("关闭", variant="secondary")
        # Non-editable registration result output
        reg_output = gr.Textbox(label="结果", interactive=False)

    # Conversation history modal, hidden by default
    with gr.Column(visible=False, elem_classes="modal") as history_modal:
        # Conversation history title
        gr.Markdown("### 会话历史")
        # Dropdown for choosing a previous conversation
        conv_dropdown = gr.Dropdown(label="选择历史会话", choices=["请选择历史会话"], value="请选择历史会话")
        # Button to load the selected conversation
        load_conv_button = gr.Button("加载会话", variant="primary")
        # Button to close the history modal
        close_history_button = gr.Button("关闭", variant="secondary")

    # Show the registration modal
    def show_register_modal(): return gr.update(visible=True)
    # Hide the registration modal
    def hide_register_modal(): return gr.update(visible=False)
    # Show the history modal and refresh the conversation list
    def show_history_modal(username): return gr.update(visible=True), gr.update(choices=get_conversation_list(username), value="请选择历史会话")
    # Hide the history modal
    def hide_history_modal(): return gr.update(visible=False)
    # Log out and reset all UI state
    def logout(): return False, None, None, gr.update(visible=True), gr.update(visible=False), "已退出登录", [], None, [], "创建新的聊天"
    # Update the welcome text
    def update_welcome_text(username): return gr.update(value=f"### 欢迎，{username}")
    # Update the title display
    def update_title_display(title): return gr.update(value=f"## {title}")

    # Bind the register button to show the registration modal
    register_button.click(show_register_modal, None, register_modal)
    # Bind the close button to hide the registration modal
    close_button.click(hide_register_modal, None, register_modal)
    # Bind registration submission
    reg_button.click(register, [reg_username, reg_password], reg_output)

    # Bind login submission
    login_button.click(
        login, [login_username, login_password], [logged_in, current_user, current_user_id, current_conversation, login_output]
    ).then(
        # Toggle page visibility based on login state
        lambda logged: (gr.update(visible=not logged), gr.update(visible=logged)), [logged_in], [login_page, chat_page]
    ).then(
        # Update the welcome text
        update_welcome_text, [current_user], welcome_text
    ).then(
        # Load the current conversation history
        lambda username, conv_id: users_db[username]["conversations"][conv_id]["history"] if username and conv_id else [],
        [current_user, current_conversation], chatbot_history
    ).then(
        # Update the conversation title
        lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天") if username and conv_id else "创建新的聊天",
        [current_user, current_conversation], conversation_title
    ).then(
        # Refresh the title display
        update_title_display, [conversation_title], title_display)

    # Bind logout handling
    logout_button.click(
        logout, None, [logged_in, current_user, current_user_id, login_page, chat_page, login_output, chatbot, current_conversation, chatbot_history, conversation_title]
    )

    # Bind the history button to show the history modal
    history_button.click(show_history_modal, [current_user], [history_modal, conv_dropdown])
    # Bind the history close button
    close_history_button.click(hide_history_modal, None, history_modal)

    # Bind new-conversation creation
    new_conv_button.click(new_conversation, [current_user], [login_output, current_conversation]
    ).then(
        # Clear the visible chat
        lambda: [], None, chatbot
    ).then(
        # Clear chat history state
        lambda: [], None, chatbot_history
    ).then(
        # Reset the conversation title
        lambda: "创建新的聊天", None, conversation_title
    ).then(
        # Refresh the title display
        update_title_display, [conversation_title], title_display)

    # Bind loading of a selected conversation
    load_conv_button.click(load_conversation, [current_user, conv_dropdown], chatbot
    ).then(
        # Update the current conversation ID
        lambda user, conv: extract_conversation_id(conv, user), [current_user, conv_dropdown], current_conversation
    ).then(
        # Update the conversation title
        lambda username, conv: users_db[username]["conversations"][extract_conversation_id(conv, username)].get("title", "创建新的聊天") if username and conv else "创建新的聊天",
        [current_user, conv_dropdown], conversation_title
    ).then(
        # Refresh the title display
        update_title_display, [conversation_title], title_display
    ).then(
        # Hide the history modal
        hide_history_modal, None, history_modal)

    # Update the persisted chat history
    def update_history(chatbot_output, history, user, conv_id):
        # Update stored history when both the user and conversation exist
        if user and conv_id: users_db[user]["conversations"][conv_id]["history"] = chatbot_output
        return chatbot_output

    # Bind the send button
    send.click(
        send_message, [message, chatbot_history, current_user_id, current_conversation, current_user], [chatbot, chatbot_history, conversation_title]
    ).then(
        # Update chat history
        update_history, [chatbot, chatbot_history, current_user, current_conversation], chatbot_history
    ).then(
        # Update the conversation title
        lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天") if username and conv_id else "创建新的聊天",
        [current_user, current_conversation], conversation_title
    ).then(
        # Refresh the title display
        update_title_display, [conversation_title], title_display
    ).then(
        # Clear the message input
        lambda: "", None, message)

    # Bind Enter-key submission on the message box
    message.submit(
        send_message, [message, chatbot_history, current_user_id, current_conversation, current_user], [chatbot, chatbot_history, conversation_title]
    ).then(
        # Update chat history
        update_history, [chatbot, chatbot_history, current_user, current_conversation], chatbot_history
    ).then(
        # Update the conversation title
        lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天") if username and conv_id else "创建新的聊天",
        [current_user, current_conversation], conversation_title
    ).then(
        # Refresh the title display
        update_title_display, [conversation_title], title_display
    ).then(
        # Clear the message input
        lambda: "", None, message)

# Launch the Gradio app when this file is run directly
if __name__ == "__main__":
    # Start the Gradio app on local port 7861
    demo.launch(server_name="127.0.0.1", server_port=7861)