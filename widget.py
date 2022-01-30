import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup

kivy.require('2.0.0')

from kivy.uix.label import Label

from kivy.config import Config
Config.set('graphics', 'resizable', False)

from kivy.uix.button import Button

from kivy.uix.floatlayout import FloatLayout

from kivy.uix.scatter import Scatter

# The TextInput widget provides a
# box for editable plain text
from kivy.uix.textinput import TextInput

# BoxLayout arranges widgets in either
# in vertical fashion that
# is one on top of another or in
# horizontal fashion that is one after another.
from kivy.uix.boxlayout import BoxLayout
from processing import test_model

# Create the App class
class NLPApp(App):

    def build(self):

        titleLabel = Label(text="Write your mock review in the box below",font_size=20, )

        # Adding the text input
        t = TextInput(font_size=20,
                      size_hint_y=None,
                      height=100)

        resultLabel = Label(text = "", font_size=30)

        btn = Button(text="Determine sentiment in the text",
                     font_size="20sp",
                     background_color=(0,0,1,1),
                     color=(1, 1, 1, 1),
                     size=(340, 80),
                     size_hint=(.2, .2),
                     pos=(240,170)
                     )



        def onClickButton(instance):
            # check if box empty
            if(len(t.text)<10):
                popup = Popup(title='Error', content=Label(text='The text inserted is either too small or empty'),
                              auto_dismiss=True,size_hint=(.4, .4), size=(100,100))
                popup.open()
                return
            else:
                prediction,score = test_model(t.text)
                # print(prediction)
                # print(score)
                resultLabel.text = "Prediction : {}   Score : {}".format(prediction, score)

        btn.bind(on_press = onClickButton)

        layout = GridLayout(cols=1, row_force_default=True,row_default_height=100)
        layout.add_widget(titleLabel ) #title top
        layout.add_widget(t) #text input
        layout.add_widget(btn) #button
        layout.add_widget(resultLabel)

        return layout

# Run the App
if __name__ == "__main__":
    NLPApp().run()