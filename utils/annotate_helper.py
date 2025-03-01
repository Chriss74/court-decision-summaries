import tkinter as tk
from tkinter import filedialog
import json
import os


class TextAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Annotation Tool")
        
        self.parent_directory=os.path.abspath(os.getcwd())

        
        self.text_box = tk.Text(root, wrap="word", width=80, height=20)
        self.text_box.pack(side="left", expand=True, fill="both")

        self.scrollbar = tk.Scrollbar(root, command=self.text_box.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.text_box.config(yscrollcommand=self.scrollbar.set)

        self.load_file_button = tk.Button(root, text="Load Text from File", command=self.load_text_from_file)
        self.load_file_button.pack(pady=10)

        self.annotation_button = tk.Button(root, text="Annotate Text", command=self.annotate_text)
        self.annotation_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack(pady=10)
        
        #class color codes
        self.class_info = {
            "other": {"color": "#ff0000"},
            "admissibility": {"color": "#00ff00"},
            "overview": {"color": "#0000ff"},
            "law": {"color": "#ffff00"},
            "interpretation": {"color": "#ff00ff"},
            "previous-ruling": {"color": "#00ffff"},
            "facts": {"color": "#800080"},
            "party-claims": {"color": "#008080"},
            "court-response": {"color": "#808000"},
            "court-ruling": {"color": "#008000"},
            "important": {"color": "#800000"},
            
        }

        # Load mappings
        annotation_mappings_path = os.path.join(os.getcwd(), "documents", "annotation_mappings.json")
        print(annotation_mappings_path)
        try:
            with open(annotation_mappings_path, "r", encoding="utf-8") as mapping_file:
                self.annotation_mappings = json.load(mapping_file)
        except:
            annotation_mappings_path=os.path.join(self.parent_directory, "documents", "annotation_mappings.json")
            with open(annotation_mappings_path, "r", encoding="utf-8") as mapping_file:
                self.annotation_mappings = json.load(mapping_file)
            

        # Drop-down list for classes
        self.selected_class_var = tk.StringVar(root)
        default_class_name = list(self.annotation_mappings.values())[0]["name"]
        self.selected_class_var.set(default_class_name)  # Set default value
        class_names = [class_info["name"] for class_info in self.annotation_mappings.values()]
        self.annotation_class_menu = tk.OptionMenu(root, self.selected_class_var, *class_names)
        self.annotation_class_menu.pack(pady=10)
        
        
    def load_text_from_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])

        if self.file_path:
            # Open and read the text from the selected file. utf-8
            with open(self.file_path, "r", encoding="utf-8") as file:
                text_content = file.read()

            # Delete previous text. Insert text
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, text_content)

    def annotate_text(self):
        selected_text = self.get_selected_text()
        

        if selected_text:
            # Get the selected class
            selected_class_name = self.selected_class_var.get()
            selected_class_id = next(class_id for class_id, class_info in self.annotation_mappings.items() if class_info["name"] == selected_class_name
)


            # Save the annotation (dictionary)
            annotation = {"class_id": selected_class_id, "text": selected_text}

            # Temporary json annotation file in order to keep track of changes
            annotations_path = os.path.join(os.getcwd(), "annotations.json")
            try:
                with open(annotations_path, "r", encoding="utf-8") as file:
                    annotations = json.load(file)
            except FileNotFoundError:
                annotations = []

            annotations.append(annotation)
            
            # Save annotations
            with open(annotations_path, "w", encoding="utf-8") as file:
                json.dump(annotations, file, indent=2)

            # background update
            tag_name = f"{selected_class_name}_tag"
            self.text_box.tag_configure(tag_name, background=self.class_info[selected_class_name]["color"])
            start_index = self.text_box.search(selected_text, "1.0", stopindex=tk.END)
            end_index = f"{start_index}+{len(selected_text)}c"
            self.text_box.tag_add(tag_name, start_index, end_index)
            self.text_box.tag_remove(tk.SEL, start_index, end_index)  # Remove selection

    def get_selected_text(self):
        # Get selected text
        start_index, end_index = self.text_box.tag_ranges(tk.SEL)
        if start_index and end_index:
            return self.text_box.get(start_index, end_index)
        else:
            return None

    def save_annotations(self):
        
        # Get annotations
        try:
            with open("annotations.json", "r", encoding="utf-8") as file:
                annotations = json.load(file)
        except FileNotFoundError:
            annotations = []

        # Save to Json
        data = {"document_id": "...........",
                "court": "ΣτΕ",
                "legal_remedy":"..........",
                "related_department": "...........",
                "annotations": annotations}
        text_with_annotations_path = os.path.join(self.parent_directory, "documents", "annotated_decisions", "text_with_annotations.json")
        with open(text_with_annotations_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        
        try:
            os.remove("annotations.json")
        except FileNotFoundError:
            pass  # File not found, nothing to delete
              

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnnotationApp(root)
    root.mainloop()
