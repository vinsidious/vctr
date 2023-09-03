class ContextManager:
    def __init__(self):
        self.globals = {}

    def get(self, name):
        return self.globals[name]

    def set(self, name, value):
        self.globals[name] = value


ctx = ContextManager()
