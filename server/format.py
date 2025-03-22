from bs4 import BeautifulSoup
import json
import tiktoken

# Define your token limit per training example
TOKEN_LIMIT = 2048
MODEL = "gpt-3.5-turbo"

def sanitize(text):
    # Escape problematic characters using json.dumps, then remove the outer quotes.
    return json.dumps(text)[1:-1]

def count_tokens(messages, model=MODEL):
    """
    Estimate token count for a list of messages in ChatML format.
    This function is adapted from OpenAI's guidance.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    # Every message follows <im_start>{role/name}\n{content}<im_end>\n
    # Here, we add a base cost per message.
    for message in messages:
        num_tokens += 4  # Base tokens for message formatting.
        # Add tokens for the role and content.
        num_tokens += len(encoding.encode(message.get("role", "")))
        num_tokens += len(encoding.encode(message.get("content", "")))
    num_tokens += 2  # Priming tokens
    return num_tokens

# Load the HTML file
with open("Discord_Chats/Direct Messages - rye bread [921646765273403474].html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Define the roles
my_username = "72"             # Replace with your assistant's username
friend_username = "rye bread"  # Replace with your friend's username
my_role = "assistant"
friend_role = "user"

# Initialize the list to store messages (without timestamp)
conversation_messages = []

# Find all message containers
msg_containers = soup.find_all("div", class_="chatlog__message")

# Iterate over each message container
for msg_container in msg_containers:
    # Extract the author's name
    author_tag = msg_container.find("span", class_="chatlog__author")
    if not author_tag:
        continue
    author = author_tag.get_text(strip=True)

    # Extract the message content
    content_tag = msg_container.find("div", class_="chatlog__content")
    if not content_tag:
        continue
    content = content_tag.get_text(separator=" ", strip=True)

    # Sanitize the content to ensure proper JSON escaping
    safe_content = sanitize(content)

    # Determine the role based on the author's username
    if author.lower() == my_username.lower():
        role = my_role
    elif author.lower() == friend_username.lower():
        role = friend_role
    else:
        continue  # Skip unknown roles

    conversation_messages.append({
        "role": role,
        "content": safe_content
    })

# Now, split conversation_messages into chunks such that each chunk's token count is <= TOKEN_LIMIT.
chunks = []
current_chunk = []
current_tokens = 0

for message in conversation_messages:
    # Estimate tokens if we add this message.
    tokens_if_added = count_tokens(current_chunk + [message])
    if tokens_if_added > TOKEN_LIMIT and current_chunk:
        # Save the current chunk and start a new one.
        chunks.append(current_chunk)
        current_chunk = [message]
    else:
        current_chunk.append(message)

# Append any remaining messages.
if current_chunk:
    chunks.append(current_chunk)

# Write each chunk as one JSON object (one line) in the output JSONL file.
with open("conversation.jsonl", "w", encoding="utf-8") as f:
    for chunk in chunks:
        conversation_obj = {"messages": chunk}
        f.write(json.dumps(conversation_obj) + "\n")
