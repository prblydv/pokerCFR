# import ipywidgets as widgets
# from IPython.display import display, clear_output
# import sys

# # Function to convert cards to HTML with colors and symbols
# def format_card(card):
#     suit_map = {
#         "C": "♣",  # Clubs
#         "D": "♦",  # Diamonds
#         "H": "♥",  # Hearts
#         "S": "♠"   # Spades
#     }
#     color_map = {
#         "C": "white",  # Clubs (white for dark mode)
#         "S": "white",  # Spades (white for dark mode)
#         "D": "red",    # Diamonds (red)
#         "H": "red"     # Hearts (red)
#     }
#     rank = card[:-1]  # Get the rank (e.g., "10" from "10D")
#     suit = card[-1]   # Get the suit (e.g., "D" from "10D")
#     suit_symbol = suit_map.get(suit, "?")
#     color = color_map.get(suit, "white")
#     return f"<span style='color: {color}; font-size: 18px;'><b>{rank}{suit_symbol}</b></span>"

# # Function to determine equity color based on its value
# def equity_color(equity):
#     if equity > 0.8:  # High equity
#         return "green"
#     elif 0.5 <= equity <= 0.8:  # Medium equity
#         return "yellow"
#     else:  # Low equity
#         return "red"

# # Initialize dynamic widgets for colorful display
# hole_cards_label = widgets.HTML(value="<b>Hole Cards: </b>")
# community_cards_label = widgets.HTML(value="<b>Community Cards: </b>")
# equity_label = widgets.HTML(value="<b>Equity: 0.00%</b>")
# state_display = widgets.VBox([hole_cards_label, community_cards_label, equity_label])

# def initialize_display():
#     """
#     Display the widgets for hole cards, community cards, and equity.
#     """
#     display(state_display)

# def update_interactive_display(equity, hole_cards, community_cards):
#     """
#     Update the display dynamically with hole cards, community cards, and equity.
#     """
#     clear_output(wait=True)

#     hole_cards_html = ", ".join([format_card(card) for card in hole_cards])
#     community_cards_html = ", ".join([format_card(card) for card in community_cards])
#     equity_html = f"<b style='color: {equity_color(equity)};'>Equity: {equity * 100:.2f}%</b>"
    
#     hole_cards_label.value = f"<b>Hole Cards: </b>{hole_cards_html}"
#     community_cards_label.value = f"<b>Community Cards: </b>{community_cards_html}"
#     equity_label.value = equity_html


# import ipywidgets as widgets
# from IPython.display import display
# import sys

# # Function to convert cards to HTML with colors and symbols
# def format_card(card):
#     suit_map = {
#         "C": "♣",  # Clubs
#         "D": "♦",  # Diamonds
#         "H": "♥",  # Hearts
#         "S": "♠"   # Spades
#     }
#     color_map = {
#         "C": "white",  # Clubs (white for dark mode)
#         "S": "white",  # Spades (white for dark mode)
#         "D": "red",    # Diamonds (red)
#         "H": "red"     # Hearts (red)
#     }
#     rank = card[:-1]  # Get the rank (e.g., "10" from "10D")
#     suit = card[-1]   # Get the suit (e.g., "D" from "10D")
#     suit_symbol = suit_map.get(suit, "?")
#     color = color_map.get(suit, "white")
#     return f"<span style='color: {color}; font-size: 18px;'><b>{rank}{suit_symbol}</b></span>"

# # Function to determine equity color based on its value
# def equity_color(equity):
#     if equity > 0.8:  # High equity
#         return "green"
#     elif 0.5 <= equity <= 0.8:  # Medium equity
#         return "yellow"
#     else:  # Low equity
#         return "red"

# # Initialize dynamic widgets for colorful display
# hole_cards_label = widgets.HTML(value="<b>Hole Cards: </b>")
# community_cards_label = widgets.HTML(value="<b>Community Cards: </b>")
# equity_label = widgets.HTML(value="<b>Equity: 0.00%</b>")
# ev_label = widgets.HTML(value="<b>EV: 0.00</b>")
# state_display = widgets.VBox([
#     hole_cards_label,
#     community_cards_label,
#     equity_label,
#     ev_label
# ])

# def initialize_display():
#     """
#     Display the widgets for hole cards, community cards, equity, and EV.
#     """
#     display(state_display)


# import ipywidgets as widgets
# from IPython.display import display, clear_output

# def update_interactive_display(equity, hole_cards, community_cards, ev, safe_bet):
#     """
#     Update the display dynamically with hole cards, community cards, equity, EV, and safe bet.
#     Change the background color based on EV value.
#     Display all information in a larger format.
#     """
#     hole_cards_html = ", ".join([format_card(card) for card in hole_cards])
#     community_cards_html = ", ".join([format_card(card) for card in community_cards])
#     equity_html = f"<b style='color: {equity_color(equity)}; font-size: 22px;'>Equity: {equity * 100:.2f}%</b>"
    
#     ev_color = "lightgreen" if ev > 0 else "lightcoral"
#     ev_html = f"<b style='background-color: {ev_color}; padding: 8px; font-size: 22px;'>ev: {ev:.2f}</b>"
    
#     safe_bet_html = f"<b style='color: white; font-size: 22px;'>sb:   {safe_bet:.2f}</b>"
    
#     # Update the labels
#     hole_cards_label.value = f"<b style='font-size: 22px;'>Hole Cards: </b>{hole_cards_html}"
#     community_cards_label.value = f"<b style='font-size: 22px;'>Community Cards: </b>{community_cards_html}"
#     equity_label.value = equity_html
#     ev_label.value = ev_html
    
#     # Create a new label for safe bet and add to the display
#     safe_bet_label = widgets.HTML(value=safe_bet_html)
    
#     # Display all widgets in a larger format
#     updated_state_display = widgets.VBox([
#         hole_cards_label,
#         community_cards_label,
#         equity_label,
#         ev_label,
#         safe_bet_label
#     ], layout=widgets.Layout(border="2px solid black", padding="10px", width="600px"))  # Increase the size of the display
    
#     # Clear previous output and display the updated display
#     clear_output(wait=True)  # Clear previous display to avoid duplication
#     display(updated_state_display)

# def update_interactive_display(equity, hole_cards, community_cards, ev, safe_bet):
#     """
#     Update the display dynamically with hole cards, community cards, equity, EV, and safe bet.
#     Change the background color based on EV value.
#     Display all information in a larger format.
#     """
#     hole_cards_html = ", ".join([format_card(card) for card in hole_cards])
#     community_cards_html = ", ".join([format_card(card) for card in community_cards])
#     equity_html = f"<b style='color: {equity_color(equity)}; font-size: 22px;'>Equity: {equity * 100:.2f}%</b>"
    
#     ev_color = "lightgreen" if ev > 0 else "lightcoral"
#     ev_html = f"<b style='background-color: {ev_color}; padding: 8px; font-size: 22px;'>EV: {ev:.2f}</b>"
    
#     safe_bet_html = f"<b style='color: blue; font-size: 22px;'>Safe Bet: {safe_bet:.2f}</b>"
    
#     # Update the labels
#     hole_cards_label.value = f"<b style='font-size: 22px;'>Hole Cards: </b>{hole_cards_html}"
#     community_cards_label.value = f"<b style='font-size: 22px;'>Community Cards: </b>{community_cards_html}"
#     equity_label.value = equity_html
#     ev_label.value = ev_html
    
#     # Create a new label for safe bet and add to the display
#     safe_bet_label = widgets.HTML(value=safe_bet_html)
    
#     # Display all widgets in a larger format
#     updated_state_display = widgets.VBox([
#         hole_cards_label,
#         community_cards_label,
#         equity_label,
#         ev_label,
#         safe_bet_label
#     ], layout=widgets.Layout(border="2px solid black", padding="10px", width="600px"))  # Increase the size of the display
    
#     # Re-display the updated display
#     display(updated_state_display)

# def update_interactive_display(equity, hole_cards, community_cards, ev):
#     """
#     Update the display dynamically with hole cards, community cards, equity, and EV.
#     Change the background color based on EV value.
#     """
#     hole_cards_html = ", ".join([format_card(card) for card in hole_cards])
#     community_cards_html = ", ".join([format_card(card) for card in community_cards])
#     equity_html = f"<b style='color: {equity_color(equity)};'>Equity: {equity * 100:.2f}%</b>"
#     ev_color = "lightgreen" if ev > 0 else "lightcoral"
#     ev_html = f"<b style='background-color: {ev_color}; padding: 5px;'>EV: {ev:.2f}</b>"
    
#     hole_cards_label.value = f"<b>Hole Cards: </b>{hole_cards_html}"
#     community_cards_label.value = f"<b>Community Cards: </b>{community_cards_html}"
#     equity_label.value = equity_html
#     ev_label.value = ev_html










import ipywidgets as widgets
from IPython.display import display, clear_output

# Function to convert cards to HTML with colors and symbols
def format_card(card):
    """
    Convert a card into an HTML-formatted string with color and symbol.
    """
    suit_map = {
        "C": "♣",  # Clubs
        "D": "♦",  # Diamonds
        "H": "♥",  # Hearts
        "S": "♠"   # Spades
    }
    color_map = {
        "C": "white",  # Clubs (black)
        "S": "white",  # Spades (black)
        "D": "red",    # Diamonds (red)
        "H": "red"     # Hearts (red)
    }
    rank = card[:-1]  # Get the rank (e.g., "10" from "10D")
    suit = card[-1]   # Get the suit (e.g., "D" from "10D")
    suit_symbol = suit_map.get(suit, "?")
    color = color_map.get(suit, "black")
    return f"<span style='color: {color}; font-size: 18px;'><b>{rank}{suit_symbol}</b></span>"

# Function to determine equity color based on its value
def equity_color(equity):
    """
    Return the color to represent equity visually.
    """
    if equity > 0.8:  # High equity
        return "green"
    elif 0.5 <= equity <= 0.8:  # Medium equity
        return "yellow"
    else:  # Low equity
        return "red"

# Initialize dynamic widgets for colorful display
hole_cards_label = widgets.HTML(value="<b>Hole Cards: </b>")
community_cards_label = widgets.HTML(value="<b>Community Cards: </b>")
equity_label = widgets.HTML(value="<b>Equity: 0.00%</b>")
state_display = widgets.VBox([hole_cards_label, community_cards_label, equity_label])

def initialize_display():
    """
    Display the widgets for hole cards, community cards, and equity.
    """
    display(state_display)

def update_interactive_display(equity, hole_cards, community_cards):
    """
    Update the display dynamically with hole cards, community cards, and equity.
    """
    # Clear the notebook cell output
    clear_output(wait=True)

    # Format the new content
    hole_cards_html = ", ".join([format_card(card) for card in hole_cards])
    community_cards_html = ", ".join([format_card(card) for card in community_cards])
    equity_html = f"<b style='color: {equity_color(equity)};'>Equity: {equity * 100:.2f}%</b>"
    
    # Update the widget values with formatted content
    hole_cards_label.value = f"<b>Hole Cards: </b>{hole_cards_html}"
    community_cards_label.value = f"<b>Community Cards: </b>{community_cards_html}"
    equity_label.value = equity_html
    
    # Display the widgets again after clearing the cell
    display(state_display)
