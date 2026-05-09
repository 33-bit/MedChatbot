from openai import OpenAI

# Initialize the client with your specific ds2api details
client = OpenAI(
    api_key="sk-591a56d71bc04ac08a7fabfb2390aa76", 
    base_url="https://ds2api-s3dd.vercel.app/v1" 
)

try:
    print("Sending request to ds2api on Vercel...\n")
    
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "user", "content": "Hello! If you receive this, please reply with a short confirmation message."}
        ],
        stream=True # Streaming is enabled to show real-time typing
    )

    print("Response: ", end="")
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
            
    print("\n\n✅ Test completed successfully!")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")