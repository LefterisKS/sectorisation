# ALL LABELS: EXACT AND FUZZY

labels_sm = exact_labels.union(fuzzy_labels)

labels_sm.count()
labels_sm.groupBy('esa').count().sort('count', ascending=False).show(40, False)

# % BY NUMBER

owner_names.count() # KEEP ORIGINAL TO BE SECTORISED

labels_sm.count()

# labels_sm.show(1000, False)

# labels_sm.filter(labels_sm['esa']=='S14').count() # show(20, False)

# labels_sm.groupBy('esa').count().sort('count', ascending=False).show(40, False)

fs = labels_sm.groupBy('esa').count().toPandas().set_index('esa')
fs.sort_values(by='count', ascending=False).plot(kind='barh', legend=False)

# % BY VALUE

vals = (
    tr
    .select(['Owner_ID', 'Value_Held'])
    .withColumn('Value_Held', tr['Value_Held'].cast(DoubleType()))
    .groupBy('Owner_ID')
    .sum()
)
vals = vals.join(labels_sm.select(['Owner_ID', 'esa']), ['Owner_ID'], 'left')
vals = vals.withColumn('sectorised', F.when(vals['esa'].isNotNull(),1).otherwise(0))
vals.groupBy('sectorised').sum().show()

##############

labels_ml['esa'].value_counts() #.plot(kind='barh')

owner_type = owner_features.select('Owner_ID', 'Owner_Type').toPandas()

labels = pd.merge(labels, owner_type, how='left', on='Owner_ID')

pd.crosstab(labels['Owner_Type'], labels['esa'])
