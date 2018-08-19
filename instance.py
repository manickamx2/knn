class Instance(object):
    def __init__(self, data):
        self.data = data
        self.label = data[0]
        self.attributes = []

    def process(self):
        iterable_data = iter(self.data)
        next(iterable_data) #skip the first element, it is the label of the instance.
        for attribute in iterable_data:
            if attribute[0] == "\"":
                attribute = attribute.replace('"', '')
                attribute = attribute.rstrip()
                self.attributes.append(attribute)
            else:
                self.attributes.append(float(attribute))

    def getLabel(self):
        return self.label

    def getAttributes(self):
        return self.attributes

    def __repr__(self):
        return '%s %s' % (self.label, self.attributes)