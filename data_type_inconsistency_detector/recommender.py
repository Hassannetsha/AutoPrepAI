class TypeRecommender:
    def recommend(self, type_counts):
        total = sum(type_counts.values())
        for dtype, count in type_counts.items():
            if count / total >= 0.8:
                if dtype in ['numeric']:
                    return 'numeric'
                elif dtype == 'datetime':
                    return 'datetime'
                elif dtype == 'boolean':
                    return 'boolean'

        if (type_counts.get('numeric', 0)) / total >= 0.7:
            return 'numeric'
        if (type_counts.get('datetime', 0)) / total >= 0.7:
            return 'datetime'
        return 'string'
