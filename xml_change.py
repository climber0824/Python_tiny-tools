import xml.etree.ElementTree as ET

def remove_xml_field(xml_file, target_field, to_delete):
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
        
    for element in root:
    
        if element.tag == 'filename':
            element.text = element.text[:-3] + 'jpg'
    
        if element.tag in to_delete:            
            root.remove(element)

        if element:
            for ele in element:        
                if ele.tag in to_delete:
                    element.remove(ele)
                
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    
    xml_file = "./pedestrian-detection-in-hazy-weather/dataset/inria_person/PICTURES_LABELS_TRAIN/test/crop_000011.xml"    
    target_field = './annotaion'
    subelement = 'segmented'
    to_delete = ['source', 'segmented', 'pose', 'truncated', 'difficult']
    
    remove_xml_field(xml_file, target_field, to_delete)