import os  # module for OS interaction

# Display current working directory
print("\nThe current directory is", os.getcwd(), "\n")

# Ask user for their name
name = input("What is your name? ")

# Ask user if they know Python
knows_python = input("Do you know the programming language Python? (yes/no) ").lower()

# Print thank-you message with user's name
print(f"\nThank you, {name.upper()} for participating!")

# Optional: Check if the user knows Python and provide a response
if knows_python == "yes":
    print("That's great!")
elif knows_python == "no":
    print(f"No worries! {name.upper()} You can always learn.")

else:
    print("Invalid response. Please enter 'yes' or 'no'.")



# directly_path = "D:/New folder (2)"
# content = os.listdir(directly_path)
# for x in content:
#     print(x,"\n ")