We will learn

https://platform.openai.com/docs/guides/prompt-engineering?prompt-example=prompt&prompt-templates-examples=filevar

to configure env outside the git repo so it will not get pushed in git

in Python OOP: the __new__() method. This is tied closely to how objects are created, and it's essential for patterns like Singleton, immutability, metaclasses, and more.

What __new__() Does

__new__() is the real constructor in Python.

It’s called before __init__() — it actually creates the instance.

It’s a class method, which means it receives the cls (class) as its first argument.

Core Idea
Only one instance of the class can be created. Any attempt to create another will return the same object.

 Real-World Use Cases
Case	Why Singleton Helps
Config loaders	Load once, share everywhere
Database connection pools	Avoid opening multiple connections
Caching systems	Centralized memory management
Logging systems	One consistent logger across the app

✅ Use Cases
Use __call__() when...	Why
You want an object to behave like a function	Clean, expressive code
You're writing a class that wraps logic or pipelines	e.g. models, validators, transformers
You're building command-style or callable handlers	like in FastAPI, ML, or async systems

🧱 Example: ML Model Wrapper
python
Copy
Edit
class Model:
    def __call__(self, x):
        return x * 2

m = Model()
print(m(3))  # ✅ 6
This makes your object behave like a reusable function with state.




Why We Used Underscores in config_loader.py

What it’s signaling:
_instance, _config_data: these are internal variables used by the class to manage state.

_load_config(): an internal method, not meant to be called directly from outside the class.

➡️ You’re saying:

"This stuff supports the class logic, but it's not part of the public interface."